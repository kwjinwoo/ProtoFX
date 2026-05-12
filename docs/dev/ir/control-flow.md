---
schema_version: 1
doc_kind: dev
title: IR control-flow subgraph contract
summary: Normative implementation-facing contract for control-flow child subgraph representation, validation, and the current `If` plus Loop carry-state MVP boundaries in ProtoFX IR.
authority: authoritative
keywords: [ir, control-flow, subgraph, if]
source_of_truth:
  - src/protofx/ir/
  - src/protofx/importers/
  - src/protofx/emitters/
  - tests/ir/
  - tests/importer/
  - tests/parity/
  - tests/downstream/
related_docs:
  - docs/adr/0008-control-flow-subgraphs-and-if-mvp.md
  - docs/adr/0009-loop-loop-carried-state-and-while-loop-lowering.md
  - docs/dev/ir/graph-model.md
  - docs/dev/ir/contracts.md
  - docs/dev/ir/invariants.md
---

# IR Control-Flow Subgraph Contract

<!-- section:purpose -->
## Purpose

This document defines the implementation-facing contract for representing, importing, validating, and lowering
control-flow child subgraphs in ProtoFX IR.

<!-- section:scope -->
## Scope

This contract covers shared control-flow foundation rules, the end-to-end `If` milestone, and the first `Loop`
milestone for loop-carried state semantics.

It does not define `Scan`, Loop scan-output semantics, a full region IR, or general dynamic-shape propagation.

<!-- section:contract -->
## Contract

- A control-flow node must keep child subgraphs in a dedicated structural mapping separate from `Node.attributes`.
- The child-subgraph mapping is keyed by ONNX attribute name.
- A `GRAPH` attribute contributes exactly one child graph.
- A `GRAPHS` attribute contributes an ordered collection of child graphs preserving ONNX attribute order.
- Each child graph is an independent `ir.Graph` whose `parent` points at the enclosing graph.
- Graph membership, value registries, initializers, interface lists, and topological traversal remain graph-local.
- Import normalization must materialize outer-scope capture as explicit child-graph inputs before validation.
- `If` support requires strict then/else output arity agreement.
- `Loop` stores its body in `node.subgraphs["body"]`.
- `Loop` uses op-specific explicit capture normalization rather than the generic structural-only non-`If` path.
- The current `Loop` milestone requires an explicit parent `cond` input; omitted-`cond` Loop forms are unsupported.
- The normalized `Loop` body interface is ordered as iteration counter, incoming condition, loop-carried state inputs,
  and then explicit captures.
- The current `Loop` milestone supports only updated condition plus updated loop-carried state outputs from the body;
  scan outputs are unsupported in this milestone.
- The parent `Loop` node outputs correspond only to final loop-carried state values in this milestone.
- `Loop` lowering targets `torch.while_loop` through a handler-owned control-flow contract.
- Dtype and shape checks for branch outputs apply only when enough metadata is available to make the comparison
  meaningful.
- The emitter may use internal recursive child-graph helpers, but the public lowering surface remains
  `emit_graph(graph) -> torch.fx.GraphModule` and operator semantics remain handler-owned.

<!-- section:invariants -->
## Invariants

- Child graphs are structural node-owned data, not scalar normalized attributes.
- No child graph may rely on hidden outer-scope values after import normalization.
- Parent and child graphs are not flattened into one global topological or ownership space.
- Validation is recursive by graph nesting, but invariant checks remain graph-local at each graph boundary.
- `Loop` loop-carried state arity must stay consistent between the parent node contract and the body graph contract.
- Unsupported control-flow operators must not silently degrade into dataflow-only behavior.

<!-- section:failure-semantics -->
## Failure Semantics

- Invalid child-graph structure, invalid ownership, or invalid interface wiring must fail through IR validation rather
  than being deferred to the emitter.
- Unsupported ONNX control-flow forms or unsupported control-flow operators remain explicit failures rather than
  falling back to partial lowering.
- `If` branch output arity mismatches are validation failures.
- Unsupported `Loop` scan outputs are explicit failures in the current milestone.
- Unsupported omitted-`cond` Loop forms are explicit failures in the current milestone.
- Metadata mismatches that cannot be proven because the model omits information remain unknown rather than being
  guessed.

<!-- section:non-goals -->
## Non-Goals

- Defining `Scan` scanned-output semantics
- Defining `Loop` scan-output semantics
- Providing a compiler-style region or CFG IR
- Requiring full static dtype or shape knowledge for all branch outputs

<!-- section:references -->
## References

- Related code: `src/protofx/ir/`, `src/protofx/importers/_onnx.py`, `src/protofx/emitters/_fx.py`
- Related tests: `tests/ir/`, `tests/importer/`, `tests/parity/`, `tests/downstream/`
- Related ADRs: `docs/adr/0001-thin-graph-owned-ir.md`, `docs/adr/0008-control-flow-subgraphs-and-if-mvp.md`, `docs/adr/0009-loop-loop-carried-state-and-while-loop-lowering.md`
