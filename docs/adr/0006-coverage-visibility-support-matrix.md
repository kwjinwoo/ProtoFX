# ADR-0006: Coverage visibility via support matrix

- Status: Superseded by ADR-0007
- Date: 2026-04-14

## Context

ProtoFX already exposes multiple authoritative validation surfaces.

- `docs/status/OPSET_COMPATIBILITY.md` summarizes op-level registry coverage.
- `tests/models/manifests/` and `tests/models/` define and validate current reference-model parity coverage.
- `tests/downstream/` defines current downstream PyTorch-tooling compatibility coverage.

Those facts are structurally present in the repository, but they are not visible in a single at-a-glance surface for
repository readers.

That creates a visibility problem:

- readers cannot quickly tell which model families are currently covered
- readers cannot quickly tell which downstream tasks are validated for which exact models
- broad phrases such as "downstream support" risk overstating the current representative validation contract

The project needs a durable visibility surface that makes current validated coverage obvious without changing the
existing validation authority model.

## Decision

ProtoFX adopts a support-matrix visibility surface rather than a separate `modelzoo` subsystem.

- `docs/status/SUPPORT_MATRIX.md` is the detailed repository-facing summary of current validated model coverage and
  downstream-task coverage.
- The authoritative sources of truth remain `tests/models/manifests/`, `tests/models/`, and `tests/downstream/`.
- The support matrix is a derived view for discoverability. It does not own pass/fail status, asset policy, or
  compatibility guarantees.
- Coverage summaries use explicit public status vocabulary: `Validated`, `Synthetic only`,
  `Not yet model-validated`, and `Planned`.
- The visibility surface presents two complementary views: a family rollup and a model-by-task matrix.
- High-level repository entrypoints may later expose a compact snapshot that links to the detailed support matrix,
  but those entrypoints remain derived views and must not broaden the authoritative contract.
- Coverage summaries must not convert representative downstream validation into an exhaustive support claim.

## Consequences

### Benefits

- Repository readers get a single place to inspect current model-family and downstream-task coverage.
- Existing validation boundaries remain intact instead of being replaced by a new asset or tooling subsystem.
- Public support messaging becomes more precise because exact model validation is separated from synthetic-only scope.

### Costs

- The support matrix becomes another derived document that must stay synchronized with manifests and tests.
- Contributors must distinguish carefully between exact model validation and task-level synthetic coverage.
- A repository entrypoint snapshot still requires follow-up implementation outside the core validation specs.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Create a separate `modelzoo` subsystem | Frames the problem as asset storage rather than coverage visibility and suggests a broader product surface than ProtoFX currently provides |
| Only update `docs/ROADMAP.md` | Does not give repository readers an at-a-glance summary of current support |
| Use broad "supported downstream" wording without a matrix | Risks overstating representative validation as exhaustive compatibility |

## Derived Specifications

- `docs/dev/ARCHITECTURE.md`
- `docs/status/SUPPORT_MATRIX.md`
- `docs/dev/MODEL_VALIDATION.md`
- `docs/dev/DOWNSTREAM_VALIDATION.md`
- `docs/ROADMAP.md`
