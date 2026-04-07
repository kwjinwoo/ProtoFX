# Downstream Validation

This document defines the suite boundary, supported environment, execution contract, and completion
criteria for ProtoFX validation against downstream PyTorch tooling.

The architectural decision behind this policy is recorded in
`docs/adr/0005-downstream-tooling-validation-boundary.md`.
Milestone scope is tracked in `docs/ROADMAP.md`.

## Purpose

ProtoFX aims to preserve enough graph structure that emitted `torch.fx.GraphModule` objects remain usable
with downstream PyTorch tooling, not just numerically close to ONNX Runtime.

That requires a validation layer distinct from both ONNX Runtime parity and fast emission smoke tests.

- ONNX Runtime parity answers whether ProtoFX matches ONNX semantics on selected inputs.
- Emitter smoke tests answer whether ProtoFX builds structurally sane FX graphs.
- Downstream validation answers whether emitted `GraphModule` objects execute correctly through selected
  PyTorch toolchains.

These claims should not be collapsed into a single suite boundary.

## Validation Boundary

The authoritative downstream validation surface is `tests/downstream/`.

| Suite | Primary question |
|-------|------------------|
| `tests/emitter/` | Did ProtoFX emit a structurally sane and executable `GraphModule`? |
| `tests/models/` | Does ProtoFX remain numerically close to ONNX Runtime on declared reference models? |
| `tests/downstream/` | Does the emitted `GraphModule` work with selected downstream PyTorch tooling? |

`tests/downstream/` owns PyTorch-toolchain compatibility claims for:

- `torch.compile`
- `torch.export`
- FX-based quantization (`torch.ao.quantization`)
- custom FX-pass compatibility

The suite may reuse small synthetic graphs or manifest-driven reference models, but ownership of the
compatibility claim remains with `tests/downstream/`, not with the source suite that supplied the model.

## Supported Environment

The initial official support contract for `torch.compile` compatibility is intentionally narrow.

- Ubuntu CI CPU environment
- project-supported Python and Torch versions
- default `torch.compile` backend

The following are not part of the initial guarantee unless explicitly added later:

- macOS or Windows-specific behavior
- GPU execution
- non-default `torch.compile` backends
- exhaustive coverage across every supported model or operator combination

This narrow contract keeps Milestone 4 completion criteria concrete and reproducible.

## Initial Coverage

The initial `torch.compile` representative scope is expected to cover both small and model-level cases.

- fast compile smoke coverage on representative emitted graphs under `tests/downstream/`
- manifest-backed end-to-end compile validation for selected models such as SqueezeNet, ResNet18, and BERT

This is a representative gate, not an exhaustive matrix.

- In-scope representative cases must all pass before the roadmap item can be marked complete.
- Failures outside the agreed initial scope should be tracked separately rather than silently expanding the
  completion contract.

## Pass/Fail Contract

For an in-scope `torch.compile` validation case to pass:

1. `emit_graph()` must produce an eager `GraphModule` that runs successfully.
2. `torch.compile(graph_module)` must execute without backend exceptions in the supported environment.
3. Compiled outputs must be numerically close to eager outputs for the same emitted `GraphModule`.

Milestone completion does not allow known failures inside the agreed representative scope.

## Planned Suite Shape

The detailed file list may evolve, but the intended suite boundary looks like this:

```text
tests/downstream/
├── conftest.py
├── test_compile_smoke.py
└── test_compile_models.py
```

Future Milestone 4 work may extend the same suite with files dedicated to `torch.export`, quantization, or
custom FX-pass checks.

## Execution Model

The downstream suite is expected to be heavier than default fast tests and should use its own pytest marker.

Planned command surface:

```bash
pytest tests/downstream/ -m downstream_validation -v
pytest tests/downstream/test_compile_smoke.py -m downstream_validation -v
pytest tests/ -m "not model_validation and not downstream_validation" -v
```

The marker and exact command surface become part of the implementation contract when the suite is added.

## Relationship to Scripts

Helper tooling may exist for tasks such as reproducing a compile failure, running focused local probes, or
preparing reusable artifacts.

That tooling is useful, but it is not the architectural center of downstream validation.

ProtoFX treats the pytest suite as authoritative.
Scripts are support machinery; they are not the source of truth for compatibility claims or roadmap
completion.
