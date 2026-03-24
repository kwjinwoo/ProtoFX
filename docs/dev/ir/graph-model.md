# IR Graph Model

This document defines graph ownership and mutation responsibilities in ProtoFX IR.

## Ownership Model

ProtoFX uses a graph-owned model rather than distributing structural invariants across `Node` and `Value`
constructors.

- `ir.Graph` owns node membership, value membership, topological order, and use-def consistency.
- `ir.Node` and `ir.Value` remain mutable entities, but structural updates flow through graph-aware APIs.
- `ir.TensorType` remains immutable and is replaced rather than mutated in place.

## Read-Only Structural Views

The public API stays convenient while ownership remains centralized.

- `Value.producer` and `Value.users` are read-only properties backed by graph-managed private state.
- `Node.inputs` and `Node.outputs` are read-only properties returning tuple snapshots backed by graph-managed
  private state.

This prevents external code from bypassing graph-managed rewiring rules.

## Node Model

`ir.Node` represents one normalized operation.

Expected field shape:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Stable internal identifier |
| `op_type` | `str` | ONNX operator type |
| `inputs` | `list[Value]` | Ordered input values |
| `outputs` | `list[Value]` | Ordered output values |
| `domain` | `str` | ONNX domain |
| `opset_version` | `int | None` | Node opset version |
| `attributes` | `dict[str, AttributeValue]` | Normalized attributes |
| `name` | `str | None` | Original ONNX node name |

`AttributeValue` is the normalized Python-native attribute space used after import.

## Graph Model

`ir.Graph` is the structural owner of all `Node` and `Value` instances.

Expected state:

| Field | Type | Description |
|-------|------|-------------|
| `inputs` | `list[Value]` | Ordered graph inputs |
| `outputs` | `list[Value]` | Ordered graph outputs |
| `nodes` | `list[Node]` | Nodes in topological order |

Internal registries and auto-ID counters remain private implementation details.

## Construction APIs

- `add_input(tensor_type, *, name=None)`
- `make_node(op_type, inputs, output_types, *, domain="", opset_version=None, attributes=None, name=None)`

These APIs create values and nodes as a single graph-level operation so producer and user links stay
consistent.

## Mutation APIs

All graph mutations go through `Graph` methods.

- `set_node_inputs(node, new_inputs)`
- `set_value_type(value, tensor_type)`
- `set_graph_outputs(outputs)`
- `remove_node(node)`

## Analysis APIs

- `topological_sort()`
- `validate()`

Validation must enforce graph well-formedness, use-def consistency, input/output ordering, and acyclicity.
