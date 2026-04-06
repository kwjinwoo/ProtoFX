# IR API

Public API reference for `protofx.ir`.

## Module Contract

`protofx.ir` exports the graph-owned normalized intermediate representation used between ONNX import and
`torch.fx` emission.

This page documents the public import surface. The authoritative structural contracts remain in
[../dev/IR.md](../dev/IR.md) and the focused documents under `docs/dev/ir/`.

## Exported Surface

| Symbol | Kind | Purpose | Authoritative spec |
|--------|------|---------|--------------------|
| `Graph` | Class | Structural owner of nodes, values, graph inputs, graph outputs, and initializers | [../dev/ir/graph-model.md](../dev/ir/graph-model.md), [../dev/ir/contracts.md](../dev/ir/contracts.md) |
| `Node` | Class | Normalized operation record with graph-managed input and output relationships | [../dev/ir/graph-model.md](../dev/ir/graph-model.md) |
| `Value` | Class | Primary data-flow object for inputs, node outputs, constants, initializers, and sentinels | [../dev/ir/type-system.md](../dev/ir/type-system.md), [../dev/ir/graph-model.md](../dev/ir/graph-model.md) |
| `ValueKind` | Enum | Origin classification for `Value` instances | [../dev/ir/type-system.md](../dev/ir/type-system.md) |
| `TensorType` | Dataclass | Immutable tensor metadata (`dtype`, `shape`) attached to each `Value` | [../dev/ir/type-system.md](../dev/ir/type-system.md) |
| `DType` | Enum | Backend-neutral element dtype enumeration aligned with ONNX numeric values | [../dev/ir/type-system.md](../dev/ir/type-system.md) |
| `Shape` | Type alias | Tensor shape representation (`tuple[Dim, ...] | None`) | [../dev/ir/type-system.md](../dev/ir/type-system.md) |
| `Dim` | Type alias | Single dimension representation (`int | str | None`) | [../dev/ir/type-system.md](../dev/ir/type-system.md) |
| `AttributeValue` | Type alias | Normalized Python-native attribute value space for IR node attributes | [../dev/ir/graph-model.md](../dev/ir/graph-model.md), [../dev/ir/contracts.md](../dev/ir/contracts.md) |

## Graph Usage Notes

`Graph` is the only structural owner in the public IR API.

- Create graph inputs with `add_input()`.
- Create omitted optional inputs with `add_sentinel()`.
- Create inlined constants with `add_constant()`.
- Create graph initializers with `add_initializer()`.
- Create nodes and their output values with `make_node()`.
- Rewire or update existing graph structure with `set_node_inputs()`, `set_value_type()`, `set_graph_outputs()`,
  and `remove_node()`.
- Use `topological_sort()` for dependency-safe traversal and `validate()` for invariant checking.

## Relationship Accessors

The public object model stays convenient without exposing raw structural mutation.

- `Node.inputs` and `Node.outputs` are read-only tuple snapshots.
- `Value.producer` and `Value.users` are graph-managed read-only views.
- `TensorType` is immutable and must be replaced, not mutated in place.

Consumers should treat those accessors as observation APIs, not as mutation hooks.

## Example

```python
from protofx.ir import DType, Graph, TensorType

graph = Graph(name="demo")
input_value = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1, 3)))
relu_node = graph.make_node(
    op_type="Relu",
    inputs=[input_value],
    output_types=[TensorType(dtype=DType.FLOAT32, shape=(1, 3))],
)
graph.set_graph_outputs([relu_node.outputs[0]])

graph.validate()
```

For architectural rationale and invariants, start from [../dev/IR.md](../dev/IR.md).
