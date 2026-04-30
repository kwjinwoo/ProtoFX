# ProtoFX Documentation

This directory is the agent-facing documentation entrypoint for ProtoFX.

The documentation tree is organized so agents can distinguish durable decisions, normative implementation
contracts, and derived status snapshots without reading every document in full.

## Authority Order

When documents disagree, use this precedence:

1. `docs/adr/` for accepted architectural decisions and rationale.
2. `docs/dev/` for implementation-facing specifications and contracts derived from accepted decisions.
3. `docs/status/` for derived status snapshots and coverage summaries.
4. `docs/ROADMAP.md` for planned scope and milestone priority.

## Directory Roles

| Directory | Role | Should contain | Should not contain |
|-----------|------|----------------|--------------------|
| `docs/adr/` | Decision records | durable structural decisions, rationale, alternatives, consequences | task logs, execution checklists, current support snapshots |
| `docs/dev/` | Normative implementation contracts | invariants, boundaries, failure semantics, behavior guarantees | decision history, roadmap planning, derived support summaries used as authority |
| `docs/status/` | Derived repository snapshots | support summaries, coverage snapshots, compatibility summaries | normative contracts, decision rationale |

## Common Document Format

Documents under `docs/adr/`, `docs/dev/`, and `docs/status/` should follow the same machine-readable shape.

### Front Matter

Each document should start with YAML front matter containing at least these fields:

```yaml
---
schema_version: 1
doc_kind: dev
title: IR contracts
summary: Normative implementation-facing contracts for the IR layer.
authority: authoritative
keywords: [ir, contract, invariants]
source_of_truth:
  - src/protofx/ir/
  - tests/ir/
related_docs:
  - docs/adr/0001-thin-graph-owned-ir.md
---
```

Required fields:

- `schema_version`
- `doc_kind`
- `title`
- `summary`
- `authority`
- `keywords`
- `source_of_truth`

Optional field:

- `related_docs`

### Section Markers

Every top-level section should include a stable machine-readable section marker immediately above the heading.

```md
<!-- section:contract -->
## Contract
```

Use lowercase kebab-case section ids. Keep ids stable once a document is published so fetch tools can request a
single section without reading the entire file.

## Section Schemas

Use these required top-level section ids for each document type.

### ADR

- `context`
- `decision`
- `consequences`
- `alternatives`
- `derived-docs`
- `supersession`

Template: `docs/adr/TEMPLATE.md`

### DEV

- `purpose`
- `scope`
- `contract`
- `invariants`
- `failure-semantics`
- `non-goals`
- `references`

Template: `docs/dev/TEMPLATE.md`

### STATUS

- `scope`
- `snapshot-semantics`
- `current-state`
- `source-of-truth`
- `limitations`
- `references`

Template: `docs/status/TEMPLATE.md`

If a section does not apply, keep the section and write `Not applicable.` rather than deleting it.

## Ground Rules

### `docs/adr/`

- Record only durable structural decisions and the reasoning behind them.
- Write a new ADR when the architectural direction changes in a durable way.
- Do not use ADRs as implementation logs, TODO lists, or milestone execution checklists.

### `docs/dev/`

- Record current implementation-facing contracts that code and tests are expected to satisfy.
- Prefer non-local guarantees, invariants, and boundary rules over code-discoverable API inventories.
- When implementation changes invalidate a normative contract, update `docs/dev/` to reflect the new contract.

### `docs/status/`

- Record derived snapshots that summarize current repository state.
- Always point back to tests, manifests, code, or generators in `source_of_truth`.
- Never treat a status document as authoritative over ADRs, specs, tests, or manifests.

## Start Here

- Read `docs/dev/ARCHITECTURE.md` first for the top-level system model and documentation boundaries.
- Read `docs/adr/README.md` first when the question is why a structural decision was made.
- Read `docs/dev/IR.md` first for IR contracts, graph ownership, or validation boundaries.
- Read `docs/dev/PUBLIC_API.md` first for the current public Python API surface.
- Read `docs/dev/MODEL_VALIDATION.md` first for model-validation rules.
- Read `docs/dev/DOWNSTREAM_VALIDATION.md` first for downstream-tooling validation rules.
- Read `docs/status/SUPPORT_MATRIX.md` first for a representative validation snapshot.
- Read `docs/status/OPSET_COMPATIBILITY.md` first for the generated op-level compatibility snapshot.
- Read `docs/ROADMAP.md` first for planned rather than guaranteed work.

## Usage Rule

Do not treat a summary page as broader authority than the tests, manifests, generators, or ADRs it links to.
