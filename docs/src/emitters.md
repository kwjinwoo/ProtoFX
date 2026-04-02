# Emitters API

Public API reference for `protofx.emitters`.

## Module Contract

`protofx.emitters` is the FX-facing boundary of ProtoFX. It consumes normalized IR and lowers it into an
executable `torch.fx.GraphModule`.

For the authoritative pipeline contract, see [../dev/ir/contracts.md](../dev/ir/contracts.md).

## Public API

### `emit_graph(graph: protofx.ir.Graph) -> torch.fx.GraphModule`

Convert a normalized `protofx.ir.Graph` into a `torch.fx.GraphModule`.

#### Parameters

| Name | Type | Description |
|------|------|-------------|
| `graph` | `protofx.ir.Graph` | Validated normalized IR, typically produced by `protofx.importers.import_model()`. |

#### Returns

A `torch.fx.GraphModule` whose structure mirrors the IR graph.

The emitted module has the following externally visible behavior:

- graph inputs are emitted as `torch.fx.Graph.placeholder()` nodes in IR input order
- initializers are registered as buffers on the root module and referenced through `get_attr()`
- constants and late materialized initializer values are emitted lazily when first needed
- IR node outputs are mapped back to FX nodes by output slot position
- `forward()` returns a tuple whose arity matches `graph.outputs`; single-output graphs therefore still return a one-element tuple

#### Emission Behavior

- Uses `graph.topological_sort()` as the authoritative node-ordering API.
- Converts `ValueKind.SENTINEL` inputs into `None` entries in the op handler argument list.
- Dispatches operator-specific lowering through `protofx.ops.dispatch_op(node.op_type)`.
- Sanitizes ONNX-derived names before using them as FX placeholder or buffer names by replacing `.`, `/`, `::`, and `-` with `_`; names that start with a digit are prefixed with `_`.
- Appends numeric suffixes when sanitized buffer names collide.
- Verifies that every op handler returns exactly one `torch.fx.Node` per IR output.

#### Raises

| Exception | When |
|-----------|------|
| `ValueError` | The graph contains a cycle, an input or output value cannot be resolved during emission, or an op handler returns the wrong number of outputs. |
| `NotImplementedError` | No handler is registered for an IR node's `op_type`. |

#### Example

```python
import onnx

from protofx.emitters import emit_graph
from protofx.importers import import_model

model = onnx.load("model.onnx")
graph = import_model(model)
graph_module = emit_graph(graph)

(result,) = graph_module(input_tensor)
graph_module.graph.print_tabular()
```

## Op Handler Interface

`emit_graph()` delegates each IR node to a registered op handler with the following signature:

```python
def handler(
    node: protofx.ir.Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    ...
```

Each returned FX node is matched to the corresponding IR output by index. Multi-output handlers are supported,
but the returned list length must exactly match `len(node.outputs)`.
