---
schema_version: 1
doc_kind: adr
title: ADR-0010: Scan MVP state, scanned-output, and while_loop lowering
summary: Define the first Scan milestone around normalized state and scanned-output interfaces, explicit capture normalization, default axis-0 forward semantics, and torch.while_loop as the primary lowering target.
authority: authoritative
keywords: [architecture, ir, control-flow, scan]
source_of_truth:
  - docs/adr/0010-scan-mvp-state-scanned-output-and-while-loop-lowering.md
related_docs:
  - docs/dev/IR.md
  - docs/dev/ir/control-flow.md
  - docs/dev/ir/contracts.md
  - docs/ROADMAP.md
decision_status: accepted
decision_date: 2026-05-14
---

# ADR-0010: Scan MVP state, scanned-output, and while_loop lowering

<!-- section:context -->
## Context

ADR-0008 established structural child-subgraph control flow and ADR-0009 defined the first `Loop` milestone, but
both decisions explicitly left `Scan` out of scope.

- Milestone 9 still includes an open roadmap item for `Scan` support plus representative validation coverage.
- `Scan` introduces a durable contract question that differs from `Loop`: parent outputs include both final state
  values and accumulated scanned outputs.
- The current project environment exposes `torch.while_loop` but does not expose a stable `torch.scan`-style higher
  order primitive, so ProtoFX cannot rely on a direct scan intrinsic as its architecture baseline.
- Without a decision, importer normalization, IR validation, and FX lowering would have to guess the ordering and
  ownership of state values, per-step scan slices, explicit captures, and accumulated scanned outputs.

<!-- section:decision -->
## Decision

ProtoFX adds a first `Scan` milestone focused on default forward axis-0 semantics with explicit state and scanned-output
contracts.

- `Scan` remains represented as a normal IR node with its body stored as a structural child graph in
  `node.subgraphs["body"]`. ProtoFX does not introduce a region or CFG IR for this milestone.
- The parent `Scan` node preserves normalized ordered inputs of:
  1. state initial values
  2. scan inputs
  3. explicit captures
- The parent `Scan` node outputs are ordered as:
  1. final state outputs
  2. scanned outputs
- The normalized `Scan` body interface is ordered as:
  1. incoming state values
  2. per-step scan-slice inputs
  3. explicit captures
- The normalized `Scan` body outputs are ordered as:
  1. updated state values
  2. per-step scan outputs
- The milestone supports plural arity for state values, scan inputs, and scan outputs. These remain ordered families
  rather than being narrowed to single-value special cases.
- `Scan` uses op-specific explicit capture normalization. Outer-scope values referenced by the body and not supplied by
  the formal ONNX `Scan` interface are materialized as explicit child-graph inputs and as additional normalized
  parent-node inputs after the ONNX formal inputs.
- The first `Scan` milestone supports only default forward axis-0 semantics for scan inputs and scan outputs.
  Non-default `scan_input_axes`, `scan_input_directions`, `scan_output_axes`, and `scan_output_directions` remain
  unsupported in this milestone.
- Each body scan output represents one per-iteration slice. Parent scanned outputs are assembled in iteration order
  along axis 0 under the default semantics.
- `torch.while_loop` is the primary lowering target. Lowering remains handler-owned through `@register_op("Scan")`;
  the emitter may reuse internal child-graph helpers but does not absorb Scan semantics into emitter core logic.
- The emitted loop state for `torch.while_loop` is conceptually
  `(iteration_counter, current_state_values..., accumulated_scanned_outputs...)`, with explicit captures treated as
  closed-over operands rather than loop-state elements.
- Importer-returned validated IR remains the fail-fast boundary. Scan-specific validation must reject unsupported
  non-default axes or directions, hidden captures, body-interface mismatches, and provable state or scanned-output
  metadata mismatches before emission succeeds.

<!-- section:consequences -->
## Consequences

### Benefits

- `Scan` builds on the accepted child-subgraph architecture without introducing a heavier control-flow IR.
- Default axis-0 forward semantics keep the first Scan milestone concrete and compatible with the project's existing
  downstream validation boundary.
- Using ordered state and scanned-output families prevents the first Scan milestone from overfitting to trivial
  single-input or single-output cases.

### Costs

- `Scan` adds a second operator-specific control-flow contract beyond the shared structural rules and the Loop MVP.
- `torch.while_loop`-based lowering for scanned outputs is more complex than a hypothetical direct `torch.scan`
  intrinsic would be.
- Non-default axes and directions remain explicit unsupported cases until later milestone work lands.

<!-- section:alternatives -->
## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Support the full ONNX `Scan` surface in the first milestone | Too broad for Milestone 9 and too risky without a narrower validation contract |
| Use Python-level list accumulation or eager-only fallback semantics | Weakens downstream-tooling goals and does not provide a stable higher-order lowering contract |
| Restrict the first milestone to a single state value and single scanned output | Makes the ADR too narrow and leaves obvious ordered-family behavior undefined |
| Keep capture references implicit inside the body graph | Hides dependencies and weakens fail-fast validation |

<!-- section:derived-docs -->
## Derived Documents

- `docs/adr/README.md`
- `docs/dev/IR.md`
- `docs/dev/ir/control-flow.md`
- `docs/dev/ir/contracts.md`
- `docs/ROADMAP.md`

<!-- section:supersession -->
## Supersession

Not applicable.
