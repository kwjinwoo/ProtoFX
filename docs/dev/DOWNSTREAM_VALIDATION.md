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
- PT2E quantization (`torchao.quantization.pt2e`)
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

### PT2E Quantization

The official support contract for PT2E quantization uses the same narrow environment as `torch.compile`.

- Ubuntu CI CPU environment
- project-supported Python and Torch versions
- `torchao.quantization.pt2e` (`prepare_pt2e` / `convert_pt2e`) with a symmetric int8 static quantization config
- custom `Quantizer` subclass using `torchao.quantization.pt2e.quantizer` primitives

The following are not part of the initial guarantee:

- dynamic quantization or quantization-aware training (QAT)
- per-layer quantization overrides or non-default quantization configs
- exhaustive coverage across every supported model or operator combination

## Initial Coverage

### `torch.compile`

The initial `torch.compile` representative scope covers both small and model-level cases.

- fast compile smoke coverage on representative emitted graphs under `tests/downstream/`
- manifest-backed end-to-end compile validation for selected models such as SqueezeNet, ResNet18, and BERT

### PT2E Quantization

The initial PT2E quantization representative scope covers both small and model-level cases.

- fast quantization smoke coverage on representative emitted graphs (Conv, MatMul, Add+Relu)
- manifest-backed end-to-end quantization survival for SqueezeNet

### `torch.export`

The initial `torch.export` round-trip representative scope covers small synthetic graphs only.

- fast export round-trip smoke coverage on representative emitted graphs (Relu, Add+Relu, MatMul,
  Conv, LayerNorm, multi-op Relu+Sigmoid)

### Custom FX Pass

The initial custom FX pass representative scope covers small synthetic graphs only.

- Standard library FX pass (`torch.fx.passes.shape_prop.ShapeProp`) applied to representative emitted
  graphs (Relu, Add+Relu, Conv, MatMul)
- Custom node-replacement pass (Relu → LeakyRelu) applied to representative emitted graphs (Relu,
  Add+Relu)

All scopes are representative gates, not exhaustive matrices.

- In-scope representative cases must all pass before the roadmap item can be marked complete.
- Failures outside the agreed initial scope should be tracked separately rather than silently expanding the
  completion contract.

## Pass/Fail Contract

### `torch.compile`

For an in-scope `torch.compile` validation case to pass:

1. `emit_graph()` must produce an eager `GraphModule` that runs successfully.
2. `torch.compile(graph_module)` must execute without backend exceptions in the supported environment.
3. Compiled outputs must be numerically close to eager outputs for the same emitted `GraphModule`.

### PT2E Quantization

For an in-scope PT2E quantization validation case to pass:

1. `emit_graph()` must produce an eager `GraphModule` that runs successfully.
2. `torch.export.export(graph_module, inputs).module()` must produce an exported `GraphModule`.
3. `prepare_pt2e(exported_gm, quantizer)` must complete without exceptions.
4. A calibration forward pass on the prepared model must complete without exceptions.
5. `convert_pt2e(prepared)` must complete without exceptions.
6. A forward pass on the converted (quantized) model must complete without exceptions.
7. Output shapes of the quantized model must match the eager model's output shapes.

Numerical closeness between eager and quantized outputs is **not** part of the pass/fail contract because
post-training quantization intentionally reduces precision.

### `torch.export`

For an in-scope `torch.export` round-trip validation case to pass:

1. `emit_graph()` must produce an eager `GraphModule` that runs successfully.
2. `torch.export.export(graph_module, inputs).module()` must produce an exported `GraphModule`.
3. A forward pass on the exported module must complete without exceptions.
4. Exported outputs must be numerically close to eager outputs for the same emitted `GraphModule`.

### Custom FX Pass

For an in-scope custom FX pass validation case to pass:

1. `emit_graph()` must produce an eager `GraphModule` that runs successfully.
2. The FX pass must execute on the emitted `GraphModule` without exceptions.
3. A forward pass on the transformed `GraphModule` must complete without exceptions.
4. Output shapes of the transformed model must match the eager model's output shapes.

Numerical closeness between pre-pass and post-pass outputs is **not** part of the pass/fail contract
because node-replacement passes intentionally alter graph semantics.

Milestone completion does not allow known failures inside the agreed representative scope.

## Planned Suite Shape

The detailed file list may evolve, but the intended suite boundary looks like this:

```text
tests/downstream/
├── conftest.py
├── test_compile_smoke.py
├── test_compile_models.py
├── test_export_smoke.py
├── test_quantization_smoke.py
├── test_quantization_models.py
└── test_fx_pass_smoke.py
```

Future work may extend the same suite with model-level export or additional FX-pass checks.

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
