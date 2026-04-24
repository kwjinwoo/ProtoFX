# Roadmap

This document tracks the planned development milestones for ProtoFX.
Accepted architecture decisions are recorded in `docs/adr/`.
Development specifications derived from those decisions live in `docs/dev/`.
Current validated model and downstream-task visibility is summarized in `docs/dev/SUPPORT_MATRIX.md`.

> **Status key**: ✅ Done · 🚧 In Progress · 📋 Planned · 💡 Under Consideration

---

## Milestone 1 — Foundation

Core infrastructure required before any op handler can be written.

The accepted IR direction for this milestone is recorded in `docs/adr/0001-thin-graph-owned-ir.md`.
Detailed IR contracts are documented in `docs/dev/IR.md` and the `docs/dev/ir/` specification set.
Milestone 1 contract alignment is recorded in `docs/adr/0003-milestone-1-ir-contract-reconciliation.md`.

| Status | Item |
|--------|------|
| ✅ | Thin normalized IR (`ir.Graph`, `ir.Node`, `ir.Value`, `ir.TensorType`) |
| ✅ | Refactor current frozen `Node` / `Value` bootstrap into graph-owned mutable IR |
| ✅ | Establish ADR-centered documentation and specification structure |
| ✅ | ONNX importer (`onnx.ModelProto` → IR) |
| ✅ | IR validation and normalization boundary (importer returns `graph.validate()`-clean IR) |
| ✅ | `torch.fx` emitter (IR → `GraphModule`) |
| ✅ | Op handler registry (`@register_op` decorator) |
| ✅ | Basic test infrastructure (`tests/ops/` fixtures) |
| ✅ | CI setup (lint, test) |

---

## Milestone 2 — Core Op Coverage

Handlers for the most common ONNX ops to support real-world models.

| Status | Item |
|--------|------|
| ✅ | Element-wise ops (Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, Abs, Neg, Exp, Log, Sqrt, Pow, Erf, Where, And, IsNaN) |
| ✅ | Tensor ops (Reshape, Transpose, Flatten, Squeeze, Unsqueeze, Concat, Slice, Identity, Cast, Expand, Gather, GatherND) |
| ✅ | Reduction ops (ReduceMean, ReduceSum, ReduceMax, ...) |
| ✅ | Linear algebra (MatMul, Gemm) |
| ✅ | Convolution (Conv, ConvTranspose) |
| ✅ | Normalization (BatchNormalization, LayerNormalization) |
| ✅ | Pooling (MaxPool, AveragePool, GlobalAveragePool) |
| ✅ | Activation (Softmax, LogSoftmax, Gelu, Elu, LeakyRelu, Selu, Celu, PRelu, HardSigmoid, HardSwish, Mish, Softplus, Softsign, ThresholdedRelu) |

---

## Milestone 3 — Model Validation

End-to-end correctness verification against reference ONNX models.

The accepted reference-model asset policy for this milestone is recorded in
`docs/adr/0004-externalized-reference-model-validation-assets.md`.
Detailed model-validation structure is documented in `docs/dev/MODEL_VALIDATION.md`.
The generated opset compatibility matrix lives in `docs/dev/OPSET_COMPATIBILITY.md`.

| Status | Item |
|--------|------|
| ✅ | Numerical parity tests vs. ONNX Runtime |
| ✅ | Manifest-driven reference-model validation infrastructure (externalized assets, cache-backed materialization) |
| ✅ | Support for standard model families (ResNet18, BERT, ViT-B/16) in the reference-model suite |
| ✅ | Opset version compatibility matrix generated from the live op registry |
| ✅ | Stabilize manifest-backed model materialization against exporter and dependency deprecation warnings (`LeafSpec`, `torch.jit.script_method`) |

---

## Milestone 4 — Downstream Integration

Verify that the output `GraphModule` works correctly with PyTorch tooling.

The accepted downstream validation boundary for this milestone is recorded in
`docs/adr/0005-downstream-tooling-validation-boundary.md`.
Detailed downstream validation structure is documented in `docs/dev/DOWNSTREAM_VALIDATION.md`.

| Status | Item |
|--------|------|
| ✅ | `torch.compile` compatibility |
| ✅ | FX-based quantization (`torch.ao.quantization`) |
| ✅ | Eliminate non-writable NumPy-backed initializer buffers during FX emission so emitted `GraphModule` buffers are warning-free and safe to reuse |
| ✅ | Migrate downstream quantization validation from deprecated `torch.ao.quantization` to `torchao.quantization.pt2e` |
| ✅ | Custom FX pass compatibility |
| ✅ | `torch.export` round-trip |

---

## Milestone 5 — Model Family Expansion

Broaden reference-model coverage beyond the initial three families.

| Status | Item |
|--------|------|
| ✅ | Coverage hub / support matrix for validated models and downstream tasks |
| ✅ | Additional vision models (ResNet50, EfficientNet-B0, MobileNetV2, MobileNetV3-Small) |
| ✅ | Additional NLP models (RoBERTa, DistilBERT) |
| ✅ | Multi-modal models (CLIP) |

---

## Milestone 6 — Legacy Opset Compatibility

Broaden default-domain opset coverage by normalizing representation-only schema differences in the importer.

| Status | Item |
|--------|------|
| ✅ | Importer normalization for default-domain opset 11-12 representation differences, starting with Squeeze and Unsqueeze axes attribute-to-input conversion |

---

## Milestone 7 — Decoder-Style Transformer Expansion

Broaden manifest-backed reference-model validation to decoder-style transformer families using the existing
`torch.export`-based materialization path, then extend representative model-level downstream coverage for validated
decoder manifests.

| Status | Item |
|--------|------|
| ✅ | Add manifest-backed `GPT2Model` reference-model validation through the `torch.export`-based materialization path, scoped to the concrete manifest-declared module class and input signature |
| 📋 | Add representative model-level downstream validation for the validated `GPT2Model` manifest under the existing downstream-validation boundary after model-validation parity is stable |

---

## Under Consideration 💡

Ideas not yet scheduled. Discuss with `@Architect` before moving to a milestone.

- Control-flow op support (If, Loop, Scan)
- Dynamic shape / symbolic shape propagation
- Plugin system for 3rd-party op domains
- ONNX model surgery utilities (graph editing before conversion)
- CLI tool (`protofx convert model.onnx`)
- PyPI release workflow
