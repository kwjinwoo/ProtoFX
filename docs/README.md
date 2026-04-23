# ProtoFX Documentation

This directory is the agent-facing documentation entrypoint for ProtoFX.

Use this page first to decide which document is authoritative for the question you are answering.

## Authority Order

When documents disagree, use this precedence:

1. `docs/adr/` for accepted architectural decisions and rationale.
2. `docs/dev/` for implementation-facing specifications derived from accepted decisions.
3. `docs/src/` for public API reference and module-level usage notes.
4. `docs/ROADMAP.md` for planned scope and milestone priority.

## Start Here

- Read `docs/dev/ARCHITECTURE.md` first if you need the top-level system model and documentation boundaries.
- Read `docs/adr/README.md` first if you need to know why a structural decision was made.
- Read `docs/dev/IR.md` first if the question is about IR contracts, graph ownership, or validation boundaries.
- Read `docs/dev/SUPPORT_MATRIX.md` first if you need a representative snapshot of current validation coverage.
- Read `docs/src/README.md` first if you need the public Python API surface.
- Read `docs/ROADMAP.md` first if you need milestone sequencing or unscheduled work.

## Question Map

| Question | Document |
|----------|----------|
| What is the architecture and which document category wins? | `docs/dev/ARCHITECTURE.md` |
| Why was this structural decision made? | `docs/adr/README.md` and the relevant ADR |
| What does the IR guarantee? | `docs/dev/IR.md` and `docs/dev/ir/` |
| What are the model-validation rules? | `docs/dev/MODEL_VALIDATION.md` |
| What are the downstream-tooling validation rules? | `docs/dev/DOWNSTREAM_VALIDATION.md` |
| What is the current representative validation snapshot? | `docs/dev/SUPPORT_MATRIX.md` |
| What is the public API surface? | `docs/src/README.md` |
| What is planned rather than guaranteed today? | `docs/ROADMAP.md` |

## Current Snapshot Links

- `docs/dev/SUPPORT_MATRIX.md` is a representative summary for quick orientation.
- `docs/dev/OPSET_COMPATIBILITY.md` is the detailed op-level compatibility matrix.
- `tests/models/manifests/`, `tests/models/`, and `tests/downstream/` remain the authoritative validation sources when exact current coverage matters.

## Documentation Areas

- `docs/adr/` records accepted architecture decisions and their rationale.
- `docs/dev/` records stable implementation-facing contracts and specification documents.
- `docs/src/` records public API reference for `protofx` modules.
- `docs/ROADMAP.md` records planned milestones and unscheduled ideas.

## Usage Rule

Do not treat a summary page as broader authority than the tests, manifests, or ADRs it links to.
