# ADR-0007: Agent-facing documentation entrypoint and representative support summary

- Status: Accepted
- Date: 2026-04-23

## Context

ProtoFX already has an ADR-centered documentation system, but agent readers still need to infer where to start.

- `docs/` has no root entrypoint document.
- `docs/dev/ARCHITECTURE.md` has accumulated both architectural explanation and navigation duties.
- `docs/dev/SUPPORT_MATRIX.md` was framed as a detailed current-state matrix, which is costly to maintain by hand
  and easy to let drift from the authoritative test suites.
- `docs/dev/MODEL_VALIDATION.md` and `docs/dev/DOWNSTREAM_VALIDATION.md` mix stable contracts with current
  suite inventory details, making specification documents age faster than the contracts they are meant to record.

ProtoFX needs documentation that remains directly managed, agent-friendly, and explicit about authority without
pretending to be an exhaustive live status database.

## Decision

ProtoFX adopts a root documentation entrypoint and a representative-summary model for current support visibility.

- `docs/README.md` is the first navigation surface for agent readers.
- `docs/dev/SUPPORT_MATRIX.md` becomes a representative summary rather than an exhaustive model-by-task matrix.
- Exact current coverage remains owned by `tests/models/manifests/`, `tests/models/`, and `tests/downstream/`.
- `docs/dev/MODEL_VALIDATION.md` and `docs/dev/DOWNSTREAM_VALIDATION.md` keep stable contracts, execution
  rules, and boundary definitions rather than current file inventories.
- `docs/dev/ARCHITECTURE.md` remains the top-level architecture specification, but it is no longer the primary
  directory index for the whole `docs/` tree.

## Consequences

### Benefits

- Agents get a single, explicit document to start from before reading deeper specifications.
- Coverage visibility remains useful without requiring a hand-maintained exhaustive matrix.
- Stable specification documents are less likely to drift because current-state inventory moves out of them.

### Costs

- Exact current coverage now requires one more hop into manifests or pytest suites.
- Contributors must maintain the distinction between representative summaries and authoritative validation sources.
- This ADR supersedes the earlier detailed support-matrix direction and requires the affected docs to be rewritten.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Keep `docs/dev/SUPPORT_MATRIX.md` as a detailed exhaustive matrix | Too much direct-maintenance cost for a document that is not authoritative in the first place |
| Keep using `docs/dev/ARCHITECTURE.md` as the implicit docs index | Forces agents to infer navigation from a document that should primarily explain architecture |
| Generate support summaries automatically | Would reduce drift, but the project currently wants directly managed documents rather than generated documentation |

## Derived Specifications

- `docs/README.md`
- `docs/adr/README.md`
- `docs/dev/ARCHITECTURE.md`
- `docs/dev/SUPPORT_MATRIX.md`
- `docs/dev/MODEL_VALIDATION.md`
- `docs/dev/DOWNSTREAM_VALIDATION.md`
