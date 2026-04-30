---
schema_version: 1
doc_kind: adr
title: ADR-XXXX: Short decision title
summary: One-paragraph summary of the decision and the architectural scope it affects.
authority: authoritative
keywords: [architecture, replace-me]
source_of_truth:
  - docs/adr/000X-short-title.md
related_docs:
  - docs/dev/ARCHITECTURE.md
decision_status: proposed
decision_date: YYYY-MM-DD
---

# ADR-XXXX: Short Decision Title

<!-- section:context -->
## Context

Describe the pressure, inconsistency, or design gap that requires a durable decision.

Include only the facts needed to understand why a decision is necessary.

<!-- section:decision -->
## Decision

State the decision directly and concretely.

- What is being adopted?
- What boundaries or rules does it introduce?
- What is explicitly in scope?

<!-- section:consequences -->
## Consequences

### Benefits

- Concrete benefit 1
- Concrete benefit 2

### Costs

- Concrete cost or trade-off 1
- Concrete cost or trade-off 2

<!-- section:alternatives -->
## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Option A | Reason |
| Option B | Reason |

<!-- section:derived-docs -->
## Derived Documents

- `docs/dev/...`
- `docs/ROADMAP.md` if milestone or scope changes are required

<!-- section:supersession -->
## Supersession

Not applicable.

## Notes

- Keep ADRs decision-oriented, not task-oriented.
- Do not use ADRs as implementation logs or temporary checklists.
- If this ADR replaces an older decision, update the old ADR to `Superseded by ADR-XXXX`.
