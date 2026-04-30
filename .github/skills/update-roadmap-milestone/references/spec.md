# Update Roadmap Milestone Spec

## Purpose

Record architect-approved follow-up work in the project's milestone planning document without turning that document
into a workboard or commit-level execution plan.

## Preconditions

- A shared architecture agreement already exists.
- The Architect has already decided whether the agreement needs an ADR.
- There is concrete milestone-scoped follow-up work to record, move, re-scope, or retire.

## Supported Operations

- Add a new top-level milestone.
- Add a new item to an existing milestone.
- Move an item between milestones when priority or dependency changes.
- Update item status when an agreement changes the planning state.
- Mark an item or milestone as superseded or removed when a newer agreement obsoletes it.

## Required Inputs

- A normalized summary of the accepted agreement.
- Whether ADR work was written, updated, or explicitly skipped.
- The intended milestone operation.
- The priority or dependency reason for the milestone change.

## Editing Rules

- Resolve the planning document path via `docs/README.md`. Today that path is `docs/ROADMAP.md`.
- Do not use `docs/status/` as planning authority; those pages are derived snapshots only.
- Preserve the document's existing status vocabulary, heading structure, and concise milestone-item wording.
- Keep milestone items outcome-oriented. Do not expand them into commit lists, task checkboxes, or user-specific notes.
- Check for near-duplicate items before inserting new text.
- If a change affects only one user's temporary execution order, do not use this skill.

## Out Of Scope

- Negotiating architecture decisions.
- Writing or updating ADR rationale.
- Commit-level implementation planning.
- User-maintained workboard or checklist management.
- Updating implementation-facing specs beyond the milestone wording itself.

## Output Contract

The user-facing summary after the skill runs should state:

- what milestone change was made
- why that change belongs in the planning document
- whether ADR work was required or skipped
- what Planner should read next
