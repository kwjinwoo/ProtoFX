# IR Pipeline Contracts

This document records the boundaries around the ProtoFX IR.

## Pipeline Boundary

The intended conversion pipeline is:

```text
onnx.ModelProto
  -> importer
  -> ir.Graph
  -> validation / analysis passes
  -> emitter
  -> torch.fx.GraphModule
```

## Importer Contract

The importer is responsible for ONNX-aware parsing and normalization.

It must:

- parse ONNX protobuf structures
- resolve opset and domain differences
- normalize attributes into Python-native values
- normalize constants, initializers, and omitted optional inputs into IR forms
- preserve Milestone 1 source provenance needed for diagnostics
- produce graph-valid IR or fail early

For Milestone 1, the preserved provenance scope is:

- `Graph.name`
- `Node.name`
- `Node.op_type`
- `Node.domain`
- `Node.opset_version`
- `Value.name`

The importer satisfies the fail-fast requirement by returning only graphs that pass `graph.validate()`.

The importer must not leak raw ONNX protobuf handling into the emitter.

## Validation and Analysis Contract

Validation targets normalized IR, not raw ONNX inputs.

Validation is responsible for:

- graph well-formedness
- producer and user consistency
- ordered interface consistency
- acyclicity and dependency-safe traversal via `Graph.topological_sort()`
- required attribute presence and normalized form
- shape and dtype constraints when enough metadata is available

Unknown metadata remains explicit and valid when the source model does not provide enough information.

## Emitter Contract

The emitter is responsible for FX-aware lowering from normalized IR.

It must:

- consume normalized `ir.Graph` structures rather than raw ONNX nodes
- build `torch.fx.Graph` and `torch.fx.GraphModule`
- delegate operator-specific lowering through the op handler registry
- keep `torch` imports lazy where practical

The emitter must not reinterpret raw ONNX protobuf details.

### Public Entry Point

```python
from protofx.emitters import emit_graph

gm: torch.fx.GraphModule = emit_graph(graph)
```

`emit_graph(graph: ir.Graph) -> torch.fx.GraphModule` is the sole public entry point.

### Emission Phases

1. **Placeholders** — each `GRAPH_INPUT` value emits an `fx_graph.placeholder()` node.
2. **Initializers** — each `INITIALIZER` value registers a buffer on the root module and emits
   `fx_graph.get_attr()`.
3. **Node dispatch** — nodes are walked in topological order. For each node:
   - Input values are resolved from the value map. `SENTINEL` inputs become `None`.
     `CONSTANT` and `INITIALIZER` values not yet emitted are lazily materialized as buffers.
   - The op handler is dispatched via `dispatch_op(node.op_type)`.
   - Handler outputs are mapped back to IR output values by slot position.
4. **Output** — graph outputs are packed as a tuple and emitted via `fx_graph.output()`.

The emitter must treat `Graph.topological_sort()` as the authoritative node-ordering API rather than assuming
that `graph.nodes` is already in dependency-safe order.

### FX API Restriction

The emitter uses only the following `torch.fx.Graph` high-level APIs:

- `graph.placeholder(name)`
- `graph.get_attr(attr_name)`
- `graph.call_function(target, args, kwargs)`
- `graph.output(result)`

### Op Handler Signature

```python
def handler(
    node: ir.Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
```

- `node`: the IR node being lowered (provides access to attributes, opset, etc.).
- `args`: FX node references matching IR node inputs. `SENTINEL` inputs are `None`.
- `fx_graph`: the FX graph under construction.
- `module`: the root `torch.nn.Module` for buffer or parameter registration.
- Returns a list of `torch.fx.Node`, one per IR node output (multi-output supported).

Handlers are registered with `@register_op("OpName")` and dispatched via `dispatch_op("OpName")`.
Unregistered ops raise `NotImplementedError`.

## Non-Goals

The IR boundary does not imply:

- a full compiler optimization framework
- backend-specific lowering policy embedded in the IR core
- aggressive graph rewriting as a Milestone 1 requirement
