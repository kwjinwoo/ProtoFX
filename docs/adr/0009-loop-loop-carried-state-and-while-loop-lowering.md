---
schema_version: 1
doc_kind: adr
title: ADR-0009: Loop loop-carried state and while_loop lowering
summary: Define the first Loop milestone around loop-carried state semantics, a normalized body interface, and torch.while_loop as the primary lowering target.
authority: authoritative
keywords: [architecture, ir, control-flow, loop]
source_of_truth:
  - docs/adr/0009-loop-loop-carried-state-and-while-loop-lowering.md
related_docs:
  - docs/dev/IR.md
  - docs/dev/ir/control-flow.md
  - docs/dev/ir/contracts.md
  - docs/ROADMAP.md
decision_status: accepted
decision_date: 2026-05-07
---

# ADR-0009: Loop loop-carried state and while_loop lowering

<!-- section:context -->
## Context

ADR-0008 established the structural child-subgraph model and proved the control-flow foundation with `If`, but it
explicitly deferred `Loop` and `Scan`.

- The repository now has accepted child-subgraph ownership, explicit capture normalization, recursive validation, and
  handler-driven `If` lowering.
- Milestone 9 item 1 requires `Loop` support under that accepted contract.
- `Loop` is materially more complex than `If` because it introduces iterative state, optional trip-count and
  condition inputs, and a body interface whose outputs feed back into later iterations.
- The current runtime exposes `torch.while_loop`, so ProtoFX can target a higher-order loop primitive instead of
  inventing a Python-level fallback contract as the architectural baseline.

Without a durable decision, importer, IR validation, and emitter behavior would have to guess how loop-carried
state, body captures, and optional control inputs should be normalized and lowered.

<!-- section:decision -->
## Decision

ProtoFX adds a first `Loop` milestone focused on loop-carried state import and emission semantics only.

- `Loop` remains represented as a normal IR node with its body stored as a structural child graph in
  `node.subgraphs["body"]`. ProtoFX does not introduce a region or CFG IR for this milestone.
- The parent `Loop` node preserves ONNX positional inputs for `M`, `cond`, and loop-carried initial values.
- The first `Loop` milestone requires an explicit `cond` input. Omitted-`cond` ONNX Loop forms remain unsupported in
  this milestone.
- Existing sentinel normalization remains the representation for omitted optional `M` in the Loop MVP.
- `Loop` uses op-specific explicit capture normalization. Outer-scope values referenced by the body and not supplied by
  the formal ONNX `Loop` interface are materialized as explicit child-graph inputs and as additional normalized
  parent-node inputs after the ONNX positional inputs.
- The normalized body interface is ordered as:
  1. iteration counter
  2. incoming loop condition
  3. loop-carried state inputs
  4. explicit captures
- The first `Loop` milestone supports only body outputs of:
  1. updated loop condition
  2. updated loop-carried state outputs
  Scan outputs remain out of scope and therefore unsupported in this milestone.
- The parent `Loop` node outputs correspond only to final loop-carried state values in this milestone.
- `torch.while_loop` is the primary lowering target. Lowering remains handler-owned through
  `@register_op("Loop")`; the emitter may reuse internal child-graph helpers but does not absorb Loop semantics into
  emitter core logic.
- The emitted loop state for `torch.while_loop` is conceptually `(iteration_counter, current_condition,
  loop_carried_values...)`. Explicit captures are normalized as parent-node inputs but are treated as closed-over
  operands rather than as loop-state elements.
- Importer-returned validated IR remains the fail-fast boundary. Loop-specific validation must reject unsupported scan
  outputs, body interface mismatches, and provable carried-state metadata mismatches before emission succeeds.

<!-- section:consequences -->
## Consequences

### Benefits

- `Loop` can build directly on the accepted child-subgraph architecture without widening ProtoFX into a heavier IR
  design.
- `torch.while_loop` aligns the first Loop lowering with downstream-tooling goals such as `torch.compile` and
  `torch.export`.
- Scoping the milestone to loop-carried state keeps the first iterative-control-flow step smaller and easier to verify
  than a full `Loop` + scan-output implementation.

### Costs

- Loop-specific validation, capture normalization, and lowering rules add a new operator contract beyond the shared
  structural control-flow rules.
- Even with omitted-`cond` forms deferred, optional control-input handling and carried-state feedback increase importer
  and emitter complexity relative to `If`.
- Scan outputs must remain explicit unsupported cases until later milestone work lands.

<!-- section:alternatives -->
## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Extend ADR-0008 instead of writing a new ADR | `Loop` was explicitly deferred there, and its state/lowering rules are durable enough to deserve their own decision record |
| Introduce a region or CFG IR for `Loop` | Too heavy for the current milestone and unnecessary for loop-carried state MVP |
| Lower `Loop` by unrolling or Python-level control flow | Weakens the downstream-tooling architecture and avoids committing to a stable higher-order lowering contract |
| Support scan outputs in the first `Loop` milestone | Makes item 1 materially broader and overlaps the separate `Scan` milestone work |
| Support omitted-`cond` Loop forms in the first milestone | Adds awkward mode-specific semantics that do not map cleanly onto the initial `torch.while_loop` contract |

<!-- section:derived-docs -->
## Derived Documents

- `docs/adr/README.md`
- `docs/dev/IR.md`
- `docs/dev/ir/control-flow.md`
- `docs/dev/ir/contracts.md`

<!-- section:supersession -->
## Supersession

Not applicable.
