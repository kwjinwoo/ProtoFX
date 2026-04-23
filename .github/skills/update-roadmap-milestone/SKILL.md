---
name: update-roadmap-milestone
description: "Use when an architecture agreement is already explicit and the Architect needs to add or update a milestone or milestone item in the project's planning document. Trigger phrases: milestone update, roadmap milestone, add milestone, update milestone item, architecture follow-up work, agreed milestone."
user-invocable: false
---

# Update Roadmap Milestone

Architect-only workflow for recording agreed follow-up work in the project's milestone planning document.

## When to Use

- The architecture discussion is already resolved.
- The Architect has already decided whether an ADR is required.
- The agreement creates or changes milestone-scoped follow-up work.

## Do Not Use

- To resolve open architecture questions.
- To write or revise ADRs.
- To create commit-level plans or execution checklists.
- To maintain a user-specific workboard.
- From Planner or Developer as a substitute for implementation planning.

## Procedure

1. Read `docs/README.md` and resolve the current milestone planning document.
2. Confirm the architecture agreement in normalized form, including whether ADR work was required or skipped.
3. Decide the operation: add a top-level milestone, add an item to an existing milestone, move an item, update an item status, or mark work superseded.
4. Read the current planning document and check for duplicates, status-vocabulary mismatches, or the wrong insertion point.
5. Edit the planning document minimally, preserving its existing milestone structure and tone.
6. Report what changed, why it belongs in the milestone planning document, and what Planner should read next.

## Notes

- Today the planning document resolved by `docs/README.md` is `docs/ROADMAP.md`.
- If the repository later renames that planning surface, update this skill and the Architect agent together.

## Reference

- [Workflow spec](./references/spec.md)
