---
schema_version: 1
doc_kind: dev
title: IR pipeline contracts
summary: Normative importer, validation, and emitter boundaries for ProtoFX IR, including recursive child-subgraph handling and handler-driven control-flow lowering.
authority: authoritative
keywords: [ir, importer, emitter, validation, control-flow]
source_of_truth:
  - src/protofx/importers/
  - src/protofx/emitters/
  - src/protofx/ops/
  - tests/importer/
  - tests/emitter/
  - tests/downstream/
related_docs:
  - docs/adr/0001-thin-graph-owned-ir.md
  - docs/adr/0005-downstream-tooling-validation-boundary.md
  - docs/adr/0008-control-flow-subgraphs-and-if-mvp.md
  - docs/dev/ir/control-flow.md
---

# IR Pipeline Contracts

<!-- section:purpose -->
## Purpose

This document records the importer, validation, and emitter boundaries around the ProtoFX IR.

<!-- section:scope -->
## Scope

This contract covers the normalized pipeline from ONNX import through IR validation to FX emission, including the
control-flow extension for child subgraphs and `If` MVP.

<!-- section:contract -->
## Contract

The intended conversion pipeline is:

```text
onnx.ModelProto
  -> importer
  -> ir.Graph
  -> validation / analysis passes
  -> emitter
  -> torch.fx.GraphModule
```

The importer is responsible for ONNX-aware parsing and normalization.

It must:

- parse ONNX protobuf structures
- resolve opset and domain differences
- normalize scalar and list-like attributes into Python-native values
- normalize control-flow `GRAPH` and `GRAPHS` attributes into dedicated child subgraphs
- normalize constants, initializers, omitted optional inputs, and outer-scope capture into explicit IR forms
- preserve source provenance needed for diagnostics
- produce graph-valid IR or fail early

The importer satisfies the fail-fast requirement by returning only graphs that pass `graph.validate()`.
The importer must not leak raw ONNX protobuf handling into the emitter.

Validation targets normalized IR, not raw ONNX inputs.

Validation is responsible for:

- graph well-formedness
- producer and user consistency
- ordered interface consistency
- acyclicity and dependency-safe traversal via `Graph.topological_sort()`
- required attribute and child-subgraph normalized form
- recursive child-graph validation
- shape and dtype constraints when enough metadata is available

Unknown metadata remains explicit and valid when the source model does not provide enough information.

The emitter is responsible for FX-aware lowering from normalized IR.

It must:

- consume normalized `ir.Graph` structures rather than raw ONNX nodes
- build `torch.fx.Graph` and `torch.fx.GraphModule`
- delegate operator-specific lowering through the op handler registry
- keep `torch` imports lazy where practical
- keep control-flow semantics handler-owned even when internal child-graph helpers are used

The emitter must not reinterpret raw ONNX protobuf details.

The public entry point remains:

```python
from protofx.emitters import emit_graph

gm: torch.fx.GraphModule = emit_graph(graph)
```

`emit_graph(graph: ir.Graph) -> torch.fx.GraphModule` is the sole public entry point.

Handlers are registered with `@register_op("OpName")` and dispatched via `dispatch_op("OpName")`.
Unregistered ops raise `NotImplementedError`.

<!-- section:invariants -->
## Invariants

- `Graph.topological_sort()` is authoritative within one graph boundary and does not flatten parent and child graphs.
- Recursive lowering support does not change the handler registry as the semantic owner of operator-specific logic.
- Branch output arity for `If` must be validated before successful emission.

<!-- section:failure-semantics -->
## Failure Semantics

- Invalid normalized IR must fail before emission rather than being deferred to FX runtime behavior.
- Unsupported control-flow operators or unsupported normalized forms remain explicit `NotImplementedError` cases.
- Proven child-graph interface mismatches are validation failures.
- Metadata that cannot be proven from the model remains unknown rather than being guessed.

<!-- section:non-goals -->
## Non-Goals

- A full compiler optimization framework
- Backend-specific lowering policy embedded in the IR core
- A flattened global traversal order across parent and child graphs
- `Loop` and `Scan` semantics in the `If` MVP milestone

<!-- section:references -->
## References

- Related code: `src/protofx/importers/_onnx.py`, `src/protofx/emitters/_fx.py`, `src/protofx/ops/_registry.py`
- Related tests: `tests/importer/`, `tests/emitter/`, `tests/parity/`, `tests/downstream/`
- Related ADRs: `docs/adr/0001-thin-graph-owned-ir.md`, `docs/adr/0005-downstream-tooling-validation-boundary.md`, `docs/adr/0008-control-flow-subgraphs-and-if-mvp.md`
