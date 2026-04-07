# ADR-0005: Downstream tooling validation boundary

- Status: Accepted
- Date: 2026-04-08

## Context

ProtoFX Milestone 4 needs to verify that emitted `torch.fx.GraphModule` objects work with downstream
PyTorch tooling such as `torch.compile`, `torch.export`, FX-based quantization, and custom FX passes.

The existing validation surfaces do not own that claim today.

- `tests/emitter/` verifies emission behavior, graph structure, and fast smoke coverage.
- `tests/parity/` verifies ONNX Runtime numerical parity on small synthetic models.
- `tests/models/` verifies manifest-driven end-to-end numerical parity on representative reference models.

Those suites answer different questions. Numerical parity does not prove downstream PyTorch tooling
compatibility, and fast emission smoke tests should not become the architectural owner of heavyweight
toolchain checks.

ProtoFX has already established that validation claims should be grounded in authoritative pytest suites
rather than ad hoc scripts. Milestone 4 therefore needs a durable boundary for where downstream
compatibility claims live, what environment they cover, and how roadmap completion is judged.

## Decision

ProtoFX assigns downstream PyTorch tooling compatibility to a dedicated pytest suite under
`tests/downstream/`.

- `tests/downstream/` is the authoritative validation surface for Milestone 4 compatibility claims.
- `tests/emitter/` remains responsible for fast emission structure and smoke behavior.
- `tests/models/` remains responsible for ONNX Runtime parity and manifest-driven reference-model
  validation, not downstream PyTorch-tooling compatibility.
- Helper tooling may exist under `scripts/` or another utility location for local reproduction,
  investigation, or cache preparation, but scripts are not authoritative for pass/fail status.
- Initial `torch.compile` compatibility is scoped to Ubuntu CI CPU execution, using the default
  `torch.compile` backend on the project's supported Python and Torch versions.
- Initial coverage starts with representative emitted graphs and selected reference models rather than the
  entire model-validation matrix.
- A downstream compatibility item is complete only when the agreed in-scope representative cases pass
  without known failures in the supported environment.

## Consequences

### Benefits

- Keeps validation authority aligned with existing project policy: pytest suites are authoritative and
  scripts stay subordinate.
- Preserves clear suite boundaries between emission smoke tests, ONNX Runtime parity, reference-model
  parity, and PyTorch downstream-tooling compatibility.
- Gives Milestone 4 a stable architectural home for later `torch.export`, FX quantization, and custom
  FX-pass validation.
- Makes roadmap completion criteria concrete enough for CI to enforce.

### Costs

- Adds another test-suite boundary and marker that contributors must maintain.
- Requires narrow documentation of the supported environment because downstream tooling behavior is
  platform- and backend-sensitive.
- Starts with representative rather than exhaustive coverage, so later expansion still needs explicit
  planning.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Use `scripts/` as the primary `torch.compile` validation surface | Conflicts with the project's validation-boundary policy and does not provide a durable roadmap completion contract |
| Add `torch.compile` checks to `tests/emitter/` | Blurs the boundary between fast structural emission checks and heavyweight downstream execution checks |
| Add `torch.compile` checks to `tests/models/` | Collapses ONNX Runtime parity and downstream PyTorch tooling compatibility into one suite |
| Declare broad cross-platform or multi-backend support immediately | Makes the first Milestone 4 contract too ambiguous and costly to enforce |

## Derived Specifications

- `docs/dev/ARCHITECTURE.md`
- `docs/dev/DOWNSTREAM_VALIDATION.md`
- `docs/dev/MODEL_VALIDATION.md`
- `docs/ROADMAP.md`
