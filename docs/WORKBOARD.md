# Workboard

This document is a user-maintained execution checklist for directing agents.

It is intentionally separate from `docs/ROADMAP.md`.

## Intended Use

- Use this document to define the next execution order you want agents to follow.
- Keep items small enough to hand to an agent in one request.
- Delete completed sections when they are no longer useful.
- This workboard reflects the current IR development status in the repository and can be rewritten as work advances.

## Non-Authority Notice

This document is not an architectural source of truth.
It must not override accepted ADRs, development specifications, or roadmap scope.

## Current Goal

Complete the Milestone 1 IR foundation so importer and emitter work can target a stable graph-owned IR.

The Milestone 1 contract alignment that narrowed provenance scope, made `topological_sort()` the authoritative
ordering view, and kept fail-fast import validation at the boundary is recorded in
`docs/adr/0003-milestone-1-ir-contract-reconciliation.md`.

## Completed Baseline

These items are already present in the current codebase and should generally not be reopened unless a later task
forces a design change.

- [x] Establish the ADR-centered documentation system and IR documentation split.
- [x] Implement `Dim`, `Shape`, and shape helper functions with tests.
- [x] Implement the backend-neutral `DType` enum with ONNX-aligned numeric values.
- [x] Implement immutable `TensorType` with tests.
- [x] Implement `ValueKind` and mutable `Value` with read-only `producer` and `users` views.
- [x] Implement mutable `Node` with graph-managed read-only `inputs` and `outputs` views.
- [x] Implement `Graph.add_input()` for graph-owned input creation.
- [x] Implement `Graph.make_node()` for graph-owned node and output construction.
- [x] Implement graph mutation helpers: `set_node_inputs()`, `set_value_type()`, and `set_graph_outputs()`.
- [x] Implement graph analysis and safety helpers: `remove_node()`, `topological_sort()`, and `validate()`.
- [x] Implement dtype bridge utilities: `onnx_dtype_to_ir()` and `ir_dtype_to_torch()`.
- [x] Add IR unit tests covering the current graph-owned mutable model.

## Current Sequence

Execute the remaining IR work in this order.

- [x] Ask `@Planner` for a commit-level plan to add explicit graph APIs for `SENTINEL`, `CONSTANT`, and `INITIALIZER` values.
- [x] Implement graph-owned construction helpers for omitted optional inputs, constants, and initializers.
- [x] Add tests for sentinel, constant, and initializer creation, ownership, and validation behavior.
- [x] Tighten `Graph.validate()` to cover graph outputs, boundary values, kind-specific producer rules, and ownership errors beyond the current structural checks.
- [x] Reconcile Milestone 1 provenance scope and node-ordering expectations with the implemented IR contract.
- [x] Ask `@Planner` for a commit-level plan for the first ONNX importer slice targeting the current IR contract.
- [x] Create the initial importer package and convert minimal `onnx.ModelProto` graphs into `ir.Graph` inputs, initializers, and nodes.
- [x] Implement constant and initializer normalization in the importer against the graph-owned IR APIs.
- [x] Add importer tests for minimal ONNX fixtures and invariant preservation.
- [x] Ask `@Planner` for a commit-level plan for the first FX emitter slice targeting the current IR contract.
- [x] Create the initial emitter package that consumes normalized `ir.Graph` and produces a minimal `torch.fx.Graph` or `GraphModule`.
- [x] Add emitter tests and one end-to-end smoke test covering ONNX -> IR -> FX on a minimal model.
- [ ] Update the importer entry point to call `graph.validate()` before returning so the fail-fast boundary documented for Milestone 1 is actually enforced.
- [ ] After importer fail-fast validation lands, re-review the importer/emitter boundary and remove this workboard if no execution checklist is still needed.

## Deferred Until Needed

Do not prioritize these items unless a concrete milestone task requires them.

- [ ] Generic pass-manager infrastructure.
- [ ] Rewrite libraries beyond current graph mutation helpers.
- [ ] Full control-flow region ownership for `If`, `Loop`, and `Scan`.
- [ ] Symbolic shape constraint solving beyond the current `Dim` / `Shape` model.
- [ ] Plugin-domain architecture for third-party op lowering.

## Cleanup Rule

When every checklist item in this document is complete, delete the completed checklist content or remove this
document entirely.
