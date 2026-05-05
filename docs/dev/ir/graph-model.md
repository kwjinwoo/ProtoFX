---
schema_version: 1
doc_kind: dev
title: IR graph model
summary: Normative graph-ownership, node-shape, and mutation contract for ProtoFX IR, including control-flow child subgraph structure.
authority: authoritative
keywords: [ir, graph, node, ownership, control-flow]
source_of_truth:
  - src/protofx/ir/
  - tests/ir/
related_docs:
  - docs/adr/0001-thin-graph-owned-ir.md
  - docs/adr/0008-control-flow-subgraphs-and-if-mvp.md
  - docs/dev/ir/control-flow.md
---

# IR Graph Model

<!-- section:purpose -->
## Purpose

This document defines graph ownership and mutation responsibilities in ProtoFX IR.

<!-- section:scope -->
## Scope

This contract covers the structural shape of `ir.Graph` and `ir.Node`, the ownership boundary for graph-managed
relationships, and the control-flow extension that introduces child subgraphs.

<!-- section:contract -->
## Contract

ProtoFX uses a graph-owned model rather than distributing structural invariants across `Node` and `Value`
constructors.

- `ir.Graph` owns node membership, value membership, topological order, and use-def consistency.
- `ir.Node` and `ir.Value` remain mutable entities, but structural updates flow through graph-aware APIs.
- `ir.TensorType` remains immutable and is replaced rather than mutated in place.
- `Value.producer` and `Value.users` are read-only properties backed by graph-managed private state.
- `Node.inputs` and `Node.outputs` are read-only properties returning tuple snapshots backed by graph-managed private
  state.

`ir.Node` represents one normalized operation.

Expected field shape:

| Field | Description |
|-------|-------------|
| `id` | Stable internal identifier |
| `op_type` | ONNX operator type |
| `inputs` | Ordered input values |
| `outputs` | Ordered output values |
| `domain` | ONNX operator domain |
| `opset_version` | Node opset version, or `None` when unspecified |
| `attributes` | Normalized scalar and list-like attributes only |
| `subgraphs` | Dedicated child-subgraph mapping keyed by ONNX attribute name; each entry stores one child graph or an ordered child-graph collection |
| `name` | Original ONNX node name, when available |

`AttributeValue` remains the normalized Python-native attribute space used after import. Structural child graphs do
not belong to that attribute space.

`ir.Graph` is the structural owner of all `Node` and `Value` instances within one graph boundary.

Expected state:

| Field | Description |
|-------|-------------|
| `name` | Optional graph name |
| `parent` | Optional enclosing `Graph` for child-subgraph ownership |
| `inputs` | Ordered graph inputs |
| `outputs` | Ordered graph outputs |
| `initializers` | Ordered graph initializers |
| `nodes` | Nodes that belong to this graph boundary |

Internal registries and auto-ID counters remain private implementation details.

Supported graph construction APIs include:

- `add_input(*, tensor_type, name=None)`
- `add_sentinel()`
- `add_constant(*, tensor_type, data, name=None)`
- `add_initializer(*, tensor_type, data, name=None)`
- `make_node(*, op_type, inputs, output_types, domain="", opset_version=None, attributes=None, subgraphs=None, name=None, output_names=None)`

Control-flow-capable implementations may add graph-aware helpers for attaching child subgraphs, but those helpers
must preserve the same ownership rules and keep child graphs separate from scalar attributes.

All graph mutations go through `Graph` methods.

<!-- section:invariants -->
## Invariants

- Structural ownership remains centralized in `ir.Graph`.
- Child graphs remain independent graph boundaries linked only through `parent`.
- Parent and child graphs do not share one value registry or one node list.
- Control-flow structure must never bypass graph-managed ownership by storing child graphs inside scalar attributes.

<!-- section:failure-semantics -->
## Failure Semantics

- Invalid cross-graph membership, invalid ownership, or invalid use-def state must fail through graph validation.
- Implementations must not silently flatten parent and child graphs into one undifferentiated structural space.

<!-- section:non-goals -->
## Non-Goals

- Defining a full region or CFG IR
- Encoding structural child graphs as raw ONNX protobuf attributes
- Requiring `graph.nodes` to remain permanently topologically sorted after every mutation

<!-- section:references -->
## References

- Related code: `src/protofx/ir/graph.py`, `src/protofx/ir/node.py`, `src/protofx/ir/value.py`
- Related tests: `tests/ir/`
- Related ADRs: `docs/adr/0001-thin-graph-owned-ir.md`, `docs/adr/0008-control-flow-subgraphs-and-if-mvp.md`
