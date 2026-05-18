---
schema_version: 1
doc_kind: adr
title: ADR-0011: Symbolic shape propagation pipeline and validation boundary
summary: Establish symbolic shape propagation as an IR-level authoritative derived metadata layer with a defined validation boundary, comparison semantics, and scoped first-phase operator coverage.
authority: authoritative
keywords: [architecture, ir, shapes, symbolic-shapes, validation]
source_of_truth:
  - docs/adr/0011-symbolic-shape-propagation-pipeline-and-validation-boundary.md
related_docs:
  - docs/dev/IR.md
  - docs/dev/ir/contracts.md
  - docs/ROADMAP.md
decision_status: accepted
decision_date: 2026-05-18
---

# ADR-0011: Symbolic shape propagation pipeline and validation boundary

<!-- section:context -->
## Context

ProtoFX has importer-normalized tensor metadata and ONNX-provided shape hints, but it lacks a single authoritative
pipeline for deriving and validating symbolic shape relationships across nodes and control-flow boundaries.

- ONNX shape inference can provide useful initial signals, but those results are tool-dependent and not sufficient as
  a durable internal contract for IR validation and downstream lowering behavior.
- The project must preserve the current public `Dim` model (`int | str | None`) while improving internal precision for
  compatibility checks and propagation through mixed static-symbolic graphs.
- Without a clear decision, shape checks collapse into ad-hoc equality rules that cannot cleanly distinguish
  compatibility, provable incompatibility, and unknown relationships.

<!-- section:decision -->
## Decision

ProtoFX adopts symbolic shape propagation as an IR-level authoritative derived metadata layer with a fail-fast
validation boundary.

- ONNX shape inference is treated as seed input only. It may initialize propagation state but is not the authoritative
  metadata source after import.
- The propagation result attached to IR values is the authoritative derived shape metadata used by validation and
  emission preconditions.
- The public `Dim` contract remains `int | str | None`. This ADR does not introduce a public API type change.
- Internal canonical symbol identity is allowed only as an internal propagation concern for reasoning and merge
  stability; it is not exposed as a user-visible symbol model.
- Shape comparison semantics are tri-state:
  1. compatible
  2. provably incompatible
  3. unknown
- The first phase scope covers symbolic propagation for:
  - elementwise and pass-through operators
  - simple shape transforms
  - broadcast, reduction, and linear-algebra families
  - spatial operators
  - `If` propagation coverage across branch boundaries
- `Loop` and `Scan` propagation extensions are deferred to later decisions.
- Runtime dynamic lowering for shape-as-data operators is out of scope for this foundation and explicitly deferred.

<!-- section:consequences -->
## Consequences

### Benefits

- ProtoFX gains a single authoritative internal shape metadata layer instead of relying on mixed importer heuristics.
- Tri-state comparison semantics support stronger fail-fast validation while preserving explicit uncertainty when proof
  is not available.
- Public API stability is preserved while enabling richer internal symbolic reasoning.

### Costs

- Symbolic propagation introduces additional internal machinery that must remain consistent with IR validation rules.
- Some operators and control-flow forms remain intentionally deferred, so coverage is explicitly partial in the first
  phase.
- Runtime dynamic behavior for shape-as-data remains outside this decision and must be addressed by a later ADR if
  needed.

<!-- section:alternatives -->
## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Treat ONNX shape inference output as authoritative metadata | Too dependent on external inference behavior and not durable as an internal contract |
| Expand public `Dim` into a richer symbolic object model immediately | Unnecessary public API change for a foundation decision that is internal by design |
| Add runtime dynamic lowering for shape-as-data in the same decision | Couples two separate risk areas and broadens the ADR beyond a stable first boundary |
| Require binary compatible/incompatible shape checks only | Cannot represent uncertainty, leading to either false failures or silent unsound acceptance |

<!-- section:derived-docs -->
## Derived Documents

- `docs/adr/README.md`
- `docs/ROADMAP.md`

<!-- section:supersession -->
## Supersession

Not applicable.
