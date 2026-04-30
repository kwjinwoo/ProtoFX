# Project Guidelines

## Overview

ProtoFX converts ONNX models into PyTorch `torch.fx.GraphModule` objects. The goal is faithful, composable translation—not just inference parity but preserving graph structure so downstream `torch.compile`, quantization, and optimization passes work correctly.

## Code Style

- Python ≥ 3.12, type hints required on all public APIs
- Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with 120-char line limit
- Use `ruff` for linting/formatting (`ruff check` / `ruff format`)
- Prefer `match` statements over long `if/elif` chains for ONNX op dispatch
- All functions, classes, and methods must have [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- All documentation (`docs/`, docstrings, code comments) must be written in English

## Architecture

ProtoFX uses an ADR-centered documentation system.

- See [docs/README.md](../docs/README.md) first for directory roles, authority order, document templates, and section schema.
- See [docs/adr/README.md](../docs/adr/README.md) for ADR process and index.
- See [docs/dev/ARCHITECTURE.md](../docs/dev/ARCHITECTURE.md) for architecture overview, document map, and authority order.
- See [docs/dev/PUBLIC_API.md](../docs/dev/PUBLIC_API.md) for the public Python API surface and op-registry reference.
- See [docs/dev/IR.md](../docs/dev/IR.md) for the IR documentation hub.
- See [docs/dev/ir/](../docs/dev/ir/) documents for detailed IR specifications.
- See [docs/status/](../docs/status/) for derived coverage and compatibility snapshots. These are visibility aids, not normative contracts.
- See [docs/ROADMAP.md](../docs/ROADMAP.md) for planned milestones and feature status.
- See [docs/WORKBOARD.md](../docs/WORKBOARD.md) only when the user wants an execution checklist; it is not an architectural source of truth.

When documents disagree, use this precedence:

1. ADRs in `docs/adr/` for architecture decisions.
2. Specifications in `docs/dev/` for implementation-facing contracts.
3. `docs/status/` for derived snapshots only; they cannot widen ADR or spec guarantees.
4. `docs/ROADMAP.md` for milestone priority and scope.
5. `docs/WORKBOARD.md` for user-directed execution order only.

## Agent Orchestration

- Architect is the sole coordinator from initial discussion through final user report.
- Every development item follows the same required route: Architect -> Planner -> Developer -> Reviewer -> Architect.
- Planner, Developer, and Reviewer are not user-facing owners of scope or completion.
- Scope changes, architecture contradictions, and execution blockers escalate back to Architect.
- Reviewer checks both correctness and style, but must stay anchored to the Developer handoff and explicit project
  rules.

## Orchestration Artifacts

Use the current Copilot session `files/` workspace for orchestration state. Do not store this state in the git
worktree.

- Store each work item under `files/orchestration/<work-item-id>/`.
- Use JSON only for orchestration artifacts.
- Treat these artifacts as temporary orchestration state, not as source-of-truth documentation.
- Never let these artifacts override ADRs, specs, roadmap entries, or committed repository files.
- Architect owns `status.json`, chooses the `work-item-id`, and advances the phase state.
- Allowed `status.json` phases are: `discussion`, `adr_decision`, `planning`, `implementing`, `reviewing`, `rework`,
  `completed`, `blocked`.
- Architect writes `agreement.json` after shared understanding and `final.json` when the item is complete.
- Planner writes `plan.json`.
- Developer writes append-only `developer-handoff-<iteration>.json`.
- Reviewer writes append-only `review-<iteration>.json`.
- Architect decides which artifact is latest and records that in `status.json`.

## Build and Test

```bash
# Install (editable)
pip install -e ".[dev]"

# Lint & format
ruff check src/ tests/
ruff format src/ tests/

# Test
pytest tests/ -v
```

## Conventions

- Each ONNX op handler is a function decorated with `@register_op("opname")` returning `torch.fx.Node`(s)
- Prefer raising `NotImplementedError` with the op name for unsupported ops rather than silent fallback
- Test each op handler with a minimal ONNX model fixture in `tests/ops/`
- Keep `torch` imports lazy inside emitter modules to speed up import time

## Review Criteria

Reviewer style checks should anchor to explicit repository rules:

- `ruff` formatting and lint expectations
- Google-style docstrings on new or modified public APIs
- required type hints on public APIs
- 120-character line limit
- English-only project documentation and code comments
- adherence to the approved scope without unrelated cleanup

## Commit Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>
```

| Type | Usage |
|------|-------|
| `feat` | New feature (e.g., new op handler) |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or updating tests |
| `chore` | Build, CI, tooling changes |

- Scope examples: `ops`, `importer`, `emitter`, `ir`, `utils`
- Keep the description concise, lowercase, no period at the end
- Example: `feat(ops): add Conv op handler`
