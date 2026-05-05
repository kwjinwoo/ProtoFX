---
schema_version: 1
doc_kind: adr
title: ADR-0008: Control-flow subgraphs and If MVP
summary: Define how ProtoFX represents, validates, and lowers control-flow child subgraphs while scoping the first implementation milestone to shared foundation work and end-to-end `If` support.
authority: authoritative
keywords: [architecture, ir, control-flow, if]
source_of_truth:
  - docs/adr/0008-control-flow-subgraphs-and-if-mvp.md
related_docs:
  - docs/dev/IR.md
  - docs/dev/ir/control-flow.md
  - docs/dev/ir/graph-model.md
  - docs/dev/ir/contracts.md
  - docs/dev/ir/invariants.md
  - docs/ROADMAP.md
decision_status: accepted
decision_date: 2026-05-04
---

# ADR-0008: Control-flow subgraphs and If MVP

<!-- section:context -->
## Context

ProtoFX already treats future control-flow work as an expected extension of the thin graph-owned IR, but the current
repository still lacks a durable contract for how control-flow subgraphs should be represented and validated.

- `docs/dev/ARCHITECTURE.md` states that the importer recursively imports subgraphs for control flow.
- ADR-0001 and the IR invariants preserve room for future control-flow work without defining a full region model.
- The current importer normalizes only scalar and list-like ONNX attribute forms and rejects `GRAPH`-typed
  attributes.
- The current emitter and op registry are organized around handler-driven lowering and do not yet own a
  control-flow-specific lowering contract.

Starting Milestone 8 without fixing this gap would force the implementation to invent structural rules ad hoc across
the importer, IR, emitter, and tests.

<!-- section:decision -->
## Decision

ProtoFX adopts an explicit child-subgraph model for control-flow IR and scopes the first implementation milestone to
shared foundation work plus end-to-end `If` support.

- Control-flow child subgraphs are structural node data and must remain separate from normalized scalar
  `Node.attributes`.
- Each child subgraph remains an independent `ir.Graph` linked to its enclosing graph only through `Graph.parent`.
  Graph registries, topology, interfaces, and traversal remain graph-local rather than flattened across the parent
  boundary.
- Import-time normalization must convert outer-scope capture into explicit child-graph inputs so child subgraphs can
  be validated and lowered without hidden dependencies on enclosing graph state.
- `graph.validate()` is the fail-fast enforcement point for both root and child graphs. Validation recurses into child
  graphs, but each graph is checked against graph-local invariants.
- `Graph.topological_sort()` remains a graph-local API. Parent and child graphs are not flattened into one global
  topological view.
- The public emitter entry point remains `emit_graph(graph) -> torch.fx.GraphModule`. Control-flow lowering remains
  handler-driven; the emitter may expose internal child-graph helpers, but it does not take ownership of operator
  semantics away from the op handler registry.
- `If` is the first in-scope control-flow op. Its then/else branch output arity must match exactly. Dtype and shape
  validation remains conditional on available metadata rather than requiring full static knowledge.
- `Loop`, `Scan`, dynamic-shape propagation, and a full region IR remain out of scope for this decision and for the
  Milestone 8 MVP.

<!-- section:consequences -->
## Consequences

### Benefits

- Structural child graphs become first-class IR data instead of being hidden inside the scalar attribute space.
- Recursive validation and explicit capture normalization keep the importer fail-fast contract intact for control-flow
  graphs.
- The existing handler registry and public emitter boundary remain stable while still allowing recursive lowering.
- The milestone can prove the architecture with `If` without overcommitting to the much heavier iteration-state
  design needed for `Loop` and `Scan`.

### Costs

- The IR surface and validation logic grow to include nested graph structure and recursive checks.
- Import normalization must sometimes diverge from raw ONNX subgraph signatures in order to materialize explicit
  capture inputs.
- Additional specification and test coverage are required across importer, IR, parity, and downstream validation
  boundaries.

<!-- section:alternatives -->
## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Store `GRAPH` and `GRAPHS` directly inside `Node.attributes` | Mixes structural child graphs with scalar normalized attributes and makes validation and lowering rules harder to reason about |
| Flatten parent and child graphs into one registry and one traversal space | Pushes ProtoFX toward a heavier region-style IR that the project has explicitly deferred |
| Keep outer-scope capture implicit | Leaves hidden dependencies inside child graphs and weakens fail-fast validation |
| Move control-flow semantics into the emitter core instead of handlers | Breaks the existing registry-centered lowering architecture and couples FX lowering to specific operators |
| Implement `If`, `Loop`, and `Scan` under one first milestone | Makes the first milestone too broad because iteration-state semantics are materially more complex than branch-only control flow |

<!-- section:derived-docs -->
## Derived Documents

- `docs/adr/README.md`
- `docs/dev/IR.md`
- `docs/dev/ir/control-flow.md`
- `docs/dev/ir/graph-model.md`
- `docs/dev/ir/contracts.md`
- `docs/dev/ir/invariants.md`
- `docs/ROADMAP.md`

<!-- section:supersession -->
## Supersession

Not applicable.
