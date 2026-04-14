# Architecture Decision Records

This directory contains ProtoFX Architecture Decision Records (ADRs).

## Purpose

ADRs are the source of truth for accepted architecture decisions.
They exist so contributors and agents can answer the following questions consistently:

- What decision was made?
- Why was it made?
- What alternatives were rejected?
- What documents derive from that decision?

ADRs are not task logs, changelogs, or implementation checklists.

## Status Model

Each ADR should declare one of the following states near the top of the document:

- Proposed
- Accepted
- Superseded by ADR-XXXX
- Deprecated

## Naming

- Use zero-padded numeric prefixes: `0001-...`, `0002-...`
- Keep filenames short and decision-oriented
- Create a new ADR for each durable structural decision

## Template

Start new ADRs from `docs/adr/TEMPLATE.md`.

## Process

1. Discuss the architectural change.
2. Record the accepted decision in a new ADR.
3. Update the relevant specification documents under `docs/dev/`.
4. Update `docs/ROADMAP.md` if milestone priorities or scope change.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| 0001 | Thin, graph-owned normalized IR | Accepted |
| 0002 | ADR-centered documentation system | Accepted |
| 0003 | Milestone 1 IR contract reconciliation | Accepted |
| 0004 | Externalized reference-model validation assets | Accepted |
| 0005 | Downstream tooling validation boundary | Accepted |
| 0006 | Coverage visibility via support matrix | Accepted |
