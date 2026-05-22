---
schema_version: 1
doc_kind: dev
title: IR documentation hub
summary: Navigation hub for ProtoFX IR contracts, including graph ownership, invariants, pipeline boundaries, and control-flow subgraph specifications.
authority: authoritative
keywords: [ir, documentation, navigation]
source_of_truth:
  - docs/dev/ir/
  - src/protofx/ir/
  - src/protofx/importers/
  - src/protofx/emitters/
related_docs:
  - docs/adr/0001-thin-graph-owned-ir.md
  - docs/adr/0003-milestone-1-ir-contract-reconciliation.md
  - docs/adr/0008-control-flow-subgraphs-and-if-mvp.md
  - docs/adr/0009-loop-loop-carried-state-and-while-loop-lowering.md
---

# IR Documentation Hub

<!-- section:purpose -->
## Purpose

This document is the navigation hub for ProtoFX IR documentation.

<!-- section:scope -->
## Scope

This hub points readers to the authoritative IR decisions and focused implementation-facing specifications without
repeating each contract in full.

<!-- section:contract -->
## Contract

ProtoFX uses a thin normalized IR between ONNX import and `torch.fx` emission.

- `ir.Graph` owns topology, node and value registration, and use-def consistency.
- `ir.Node` and `ir.Value` remain mutable IR entities managed through graph-aware APIs.
- `ir.TensorType` remains an immutable value object.
- an internal derived-shape metadata layer is authoritative for validation and emission preconditions.
- Dependency-safe node order is obtained through `Graph.topological_sort()`.
- Control-flow child subgraphs follow the dedicated contract in `docs/dev/ir/control-flow.md`.

Use the following documents depending on the question being asked.

| Question | Document |
|----------|----------|
| Why does ProtoFX use this IR design? | `docs/adr/0001-thin-graph-owned-ir.md` |
| Why does the control-flow foundation use explicit child subgraphs, `If`, Loop carry-state scoping, and Scan scanned-output scoping? | `docs/adr/0008-control-flow-subgraphs-and-if-mvp.md`, `docs/adr/0009-loop-loop-carried-state-and-while-loop-lowering.md`, `docs/adr/0010-scan-mvp-state-scanned-output-and-while-loop-lowering.md` |
| What invariants must always hold? | `docs/dev/ir/invariants.md` |
| How are tensor metadata and value kinds modeled? | `docs/dev/ir/type-system.md` |
| How does graph ownership and mutation work? | `docs/dev/ir/graph-model.md` |
| Where is the importer / propagate / validator / emitter boundary? | `docs/dev/ir/contracts.md` |
| How are control-flow child subgraphs represented and validated? | `docs/dev/ir/control-flow.md` |

<!-- section:invariants -->
## Invariants

- This hub is index-oriented and must not become the only source of detailed IR rules.
- Focused documents under `docs/dev/ir/` remain the normative source for field shape, validation, and failure
  semantics.

<!-- section:failure-semantics -->
## Failure Semantics

When implementation details and the focused IR specifications diverge, update the focused specification documents
and record any durable structural decision in a new ADR rather than expanding this hub back into a monolithic
design document.

<!-- section:non-goals -->
## Non-Goals

- Replacing the focused IR specification documents
- Repeating full ADR rationale inside this hub
- Defining a full compiler framework or region IR

<!-- section:references -->
## References

- Related code: `src/protofx/ir/`, `src/protofx/importers/`, `src/protofx/emitters/`
- Related tests: `tests/ir/`, `tests/importer/`, `tests/parity/`, `tests/downstream/`
- Related ADRs: `docs/adr/0001-thin-graph-owned-ir.md`, `docs/adr/0003-milestone-1-ir-contract-reconciliation.md`, `docs/adr/0008-control-flow-subgraphs-and-if-mvp.md`, `docs/adr/0009-loop-loop-carried-state-and-while-loop-lowering.md`
