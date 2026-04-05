# ADR-0004: Externalized Reference-Model Validation Assets

- Status: Accepted
- Date: 2026-04-05

## Context

ProtoFX Milestone 3 expands validation beyond isolated op parity into end-to-end checks against representative
model families such as ResNet, BERT, and ViT.

The current test suite already favors small, reviewable fixtures built in code. That pattern works well for
unit tests, importer coverage, and op-level numerical parity, but it does not scale cleanly to large exported
ONNX models.

Vendoring full reference-model binaries in the repository would introduce structural costs:

- repository growth driven by generated artifacts rather than source
- low-value binary diffs that are difficult to review
- churn from exporter version, opset, and constant-folding changes
- tighter coupling between test assets and source control than the project needs

ProtoFX needs a durable boundary for where reference-model assets live, how they are declared, and how they are
materialized for local and CI validation.

## Decision

ProtoFX externalizes large reference-model validation assets.

- Reference-model validation remains a pytest-managed test suite under `tests/`, separate from small synthetic
  parity tests.
- Large ONNX model binaries and derived weight artifacts are not committed to the git repository as the normal
  validation path.
- The repository stores human-reviewable model declarations, tolerances, and metadata manifests instead of
  large generated binaries.
- Reference-model assets are materialized into a cache outside the git worktree from a canonical source or a
  reproducible export process.
- Local developer workflows may treat reference-model validation as opt-in and skippable when optional
  dependencies or cached assets are unavailable.
- Dedicated CI jobs that claim reference-model coverage must materialize declared assets and fail on missing or
  mismatched artifacts.
- Small vendored ONNX fixtures remain allowed only as narrow exceptions for bug regression coverage when
  code-generated or reproducibly exported alternatives are impractical and the review value is clear.

This decision does not require helper tooling to live in a specific directory. If helper scripts are added in
the future, they remain subordinate to the manifest-driven pytest suite rather than replacing it.

## Consequences

### Benefits

- The repository stays source-oriented instead of accumulating large generated binaries.
- Model-family validation scales without making git history and code review noisy.
- Validation boundaries stay explicit: manifests are authoritative, caches are disposable, pytest remains the
  pass/fail surface.
- Local and CI workflows can use different execution breadth without forking the validation contract.

### Costs

- ProtoFX must define and maintain manifest and cache rules for reference-model assets.
- Dedicated CI for model-family validation becomes more complex than the current lightweight parity suite.
- Some local runs will require opt-in setup, cached materialization, or graceful skipping behavior.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Commit standard-model ONNX binaries directly to the repository | Bloats the repository, produces low-signal diffs, and couples validation to generated artifacts |
| Use `scripts/` as the primary validation surface | Makes model validation look like an ad hoc tool instead of a first-class test contract |
| Restrict Milestone 3 to only synthetic code-generated models | Does not adequately exercise integrated importer and emitter behavior on representative architectures |
| Use Git LFS for vendored model assets | Reduces some git pressure but still treats large generated binaries as first-class repository content |

## Derived Specifications

- `docs/dev/ARCHITECTURE.md`
- `docs/dev/MODEL_VALIDATION.md`
- `docs/ROADMAP.md`
