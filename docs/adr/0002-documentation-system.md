# ADR-0002: ADR-Centered Documentation System

- Status: Accepted
- Date: 2026-03-22

## Context

ProtoFX documentation had begun to accumulate architectural decisions, specifications, implementation notes,
and execution order in a small number of files, especially `docs/dev/IR.md`.

That structure makes it harder for different agents to determine which document is authoritative. It also makes
architectural history difficult to preserve without turning specification documents into decision logs.

## Decision

ProtoFX adopts an ADR-centered documentation system.

- `docs/adr/` records accepted architecture decisions and their rationale.
- `docs/dev/` records technical specifications derived from accepted decisions.
- `docs/ROADMAP.md` records milestones and project-level priorities, not decision history.
- A temporary workboard may be used for user-maintained execution guidance, but it is not authoritative for
  architecture decisions and does not need to exist as a standing document.

This model intentionally separates decision records from implementation-facing specifications.

## Consequences

### Benefits

- Contributors and agents can find architectural authority in one place.
- Large documents such as the IR documentation can be split by responsibility without losing rationale.
- Roadmap discussions stop carrying architectural history that belongs in formal decision records.

### Costs

- Contributors must keep ADRs and derived specifications aligned.
- There are more documents to navigate, so the document map must stay explicit.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Keep monolithic design docs with better headings | Improves navigation but not authority separation |
| Add ADRs without restructuring specs | Leaves oversized documents and mixed responsibilities in place |
| Use roadmap as the decision log | Conflates planning priority with architectural authority |

## Derived Specifications

- `docs/dev/ARCHITECTURE.md`
- `docs/dev/IR.md`
- `docs/ROADMAP.md`
