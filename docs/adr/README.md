---
schema_version: 1
doc_kind: adr-index
title: Architecture Decision Records index
summary: Navigation and process guide for ProtoFX Architecture Decision Records.
authority: authoritative
keywords: [adr, architecture, decisions, process, index]
source_of_truth:
  - docs/adr/
  - docs/README.md
related_docs:
  - docs/adr/TEMPLATE.md
  - docs/dev/ARCHITECTURE.md
  - docs/ROADMAP.md
---

# Architecture Decision Records

This directory contains ProtoFX Architecture Decision Records (ADRs).

<!-- section:purpose -->
## Purpose

ADRs are the source of truth for accepted architecture decisions.
They exist so contributors and agents can answer the following questions consistently:

- What decision was made?
- Why was it made?
- What alternatives were rejected?
- What documents derive from that decision?

ADRs are not task logs, changelogs, or implementation checklists.

<!-- section:status-model -->
## Status Model

Each ADR should declare one of the following states near the top of the document:

- Proposed
- Accepted
- Superseded by ADR-XXXX
- Deprecated

<!-- section:naming -->
## Naming

- Use zero-padded numeric prefixes: `0001-...`, `0002-...`
- Keep filenames short and decision-oriented
- Create a new ADR for each durable structural decision

<!-- section:template -->
## Template

Start new ADRs from `docs/adr/TEMPLATE.md`.

<!-- section:process -->
## Process

1. Discuss the architectural change.
2. Record the accepted decision in a new ADR.
3. Update the relevant specification documents under `docs/dev/`.
4. Update `docs/ROADMAP.md` if milestone priorities or scope change.

<!-- section:index -->
## Index

| ADR | Title | Status |
|-----|-------|--------|
| 0001 | Thin, graph-owned normalized IR | Accepted |
| 0002 | ADR-centered documentation system | Accepted |
| 0003 | Milestone 1 IR contract reconciliation | Accepted |
| 0004 | Externalized reference-model validation assets | Accepted |
| 0005 | Downstream tooling validation boundary | Accepted |
| 0006 | Coverage visibility via support matrix | Superseded by ADR-0007 |
| 0007 | Agent-facing documentation entrypoint and representative support summary | Accepted |
| 0008 | Control-flow subgraphs and If MVP | Accepted |
| 0009 | Loop loop-carried state and while_loop lowering | Accepted |
