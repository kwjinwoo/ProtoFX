# ADR-0003: Milestone 1 IR Contract Reconciliation

- Status: Accepted
- Date: 2026-03-24

## Context

ProtoFX Milestone 1 implementation established a graph-owned mutable IR, a first ONNX importer slice, and a
first FX emitter slice. During review, several implementation-facing documents were found to be stricter than
the actual Milestone 1 design.

In particular:

- some specifications implied that `ir.Graph.nodes` must remain physically topologically ordered at all times
- provenance requirements were written more broadly than the metadata currently preserved by the codebase
- the importer contract required fail-fast behavior, but the intended Milestone 1 enforcement point was not
  explicitly defined

Leaving those mismatches in place would make Milestone 1 appear incomplete even where the implementation is
otherwise acceptable for the current roadmap scope.

## Decision

ProtoFX reconciles the Milestone 1 IR contracts to the accepted implementation shape, while keeping a strict
importer boundary.

- `ir.Graph` owns graph consistency, but `graph.nodes` is not required to remain physically topologically
  sorted after every mutation.
- `Graph.topological_sort()` is the authoritative way to obtain dependency-safe node order for validation,
  analysis, and emission.
- Milestone 1 provenance requirements are limited to metadata already preserved by the implementation:
  `Graph.name`, `Node.name`, `Node.op_type`, `Node.domain`, `Node.opset_version`, and `Value.name` when
  available.
- Milestone 1 does not require separate graph-level provenance structures or full model-level opset import
  metadata beyond the node-level opset context already carried in IR nodes.
- The importer remains responsible for fail-fast behavior, and it satisfies that responsibility by returning
  only IR that passes `graph.validate()`.

This decision is intentionally narrow. It aligns the Milestone 1 contracts with the existing implementation
shape rather than expanding IR scope.

## Consequences

### Benefits

- The documentation matches the intended Milestone 1 design instead of implying stronger guarantees than the
  project currently needs.
- Graph mutation APIs remain simpler because they do not need to maintain a permanently sorted `graph.nodes`
  list.
- Provenance scope becomes explicit and reviewable.
- The importer/emitter boundary remains strict because invalid IR must still be rejected before emission.

### Costs

- ProtoFX gives up a stronger always-sorted `graph.nodes` contract for Milestone 1.
- Provenance remains intentionally minimal, which may limit diagnostics until later milestones expand it.
- The importer required a follow-up update to enforce the documented validation boundary at the return edge.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Upgrade the implementation to satisfy the stronger original spec immediately | Too much scope for Milestone 1, especially for persistent node ordering and richer provenance metadata |
| Keep the stricter spec and only edit `docs/WORKBOARD.md` | Leaves the authoritative design documents inaccurate |
| Relax importer fail-fast requirements to best-effort import | Weakens the importer/emitter boundary and pushes invalid IR downstream |

## Derived Specifications

- `docs/dev/ARCHITECTURE.md`
- `docs/dev/IR.md`
- `docs/dev/ir/invariants.md`
- `docs/dev/ir/contracts.md`
- `docs/ROADMAP.md`
