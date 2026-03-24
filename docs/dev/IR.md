# IR (Intermediate Representation)

This document is the navigation hub for ProtoFX IR documentation.

The architectural decision behind the IR is recorded in `docs/adr/0001-thin-graph-owned-ir.md`.
The documentation system that split IR material into focused documents is recorded in
`docs/adr/0002-documentation-system.md`.
Milestone 1 contract alignment is recorded in `docs/adr/0003-milestone-1-ir-contract-reconciliation.md`.

## IR Summary

ProtoFX uses a thin normalized IR between ONNX import and `torch.fx` emission.

The IR is intentionally small. It is not a second copy of ONNX and it is not a full compiler framework.
Its job is to provide a backend-neutral graph model that is stable enough for validation, analysis, and FX
emission.

- `ir.Graph` is the owner of topology, node/value registration, and use-def consistency.
- `ir.Node` and `ir.Value` are not frozen dataclasses.
- `ir.TensorType` remains an immutable value object.
- Public convenience accessors remain part of the developer-facing API, but graph consistency is maintained by
	`ir.Graph`.
- Dependency-safe node order is obtained through `Graph.topological_sort()`, not by requiring `graph.nodes` to
	be permanently sorted.

## Document Map

Use the following documents depending on the question being asked.

| Question | Document |
|----------|----------|
| Why does ProtoFX use this IR design? | `docs/adr/0001-thin-graph-owned-ir.md` |
| What documentation model should contributors follow? | `docs/adr/0002-documentation-system.md` |
| Why do Milestone 1 IR contracts differ from some earlier draft assumptions? | `docs/adr/0003-milestone-1-ir-contract-reconciliation.md` |
| What invariants must always hold? | `docs/dev/ir/invariants.md` |
| How are tensor metadata and value kinds modeled? | `docs/dev/ir/type-system.md` |
| How does graph ownership and mutation work? | `docs/dev/ir/graph-model.md` |
| Where is the importer / validator / emitter boundary? | `docs/dev/ir/contracts.md` |

## Scope

This hub intentionally avoids repeating full decision records or full field-by-field specifications.
Those details live in the focused documents above.

## Quick Reference

- Values are the primary data-flow unit.
- `Graph` owns structural consistency.
- `Graph.topological_sort()` is the authoritative node-ordering view.
- Tensor metadata lives on `Value` via immutable `TensorType` objects.
- Importer normalizes ONNX details before emission.
- Importer returns IR only after the graph passes boundary validation.
- Validation happens against normalized IR contracts, not raw ONNX protobufs.

## Implementation Notes

The current source tree still reflects an early-stage implementation. When implementation details and the
specification diverge, update the focused spec documents and record any structural decision in a new ADR rather
than expanding this hub back into a monolithic design document.
