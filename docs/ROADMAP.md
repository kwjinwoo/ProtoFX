# Roadmap

This document tracks the planned development milestones for ProtoFX.
Accepted architecture decisions are recorded in `docs/adr/`.
Development specifications derived from those decisions live in `docs/dev/`.

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

| Status | Item |
|--------|------|
| ✅ | Numerical parity tests vs. ONNX Runtime |
| ✅ | Manifest-driven reference-model validation infrastructure (externalized assets, cache-backed materialization) |
| ✅ | Support for standard model families (ResNet18, BERT, ViT-B/16) in the reference-model suite |
| 📋 | Opset version compatibility matrix |

---

## Milestone 4 — Downstream Integration

Verify that the output `GraphModule` works correctly with PyTorch tooling.

| Status | Item |
|--------|------|
| 📋 | `torch.compile` compatibility |
| 📋 | FX-based quantization (`torch.ao.quantization`) |
| 📋 | Custom FX pass compatibility |
| 📋 | `torch.export` round-trip |

---

## Milestone 5 — Model Family Expansion

Broaden reference-model coverage beyond the initial three families.

| Status | Item |
|--------|------|
| 📋 | Additional vision models (ResNet50, EfficientNet, MobileNet, ...) |
| 📋 | Additional NLP models (GPT-2, RoBERTa, DistilBERT, ...) |
| 📋 | Multi-modal models (CLIP, ...) |
| 📋 | Op handler coverage for new model families |

---

## Under Consideration 💡

Ideas not yet scheduled. Discuss with `@Architect` before moving to a milestone.

- Opset ≤12 normalization pass in importer (e.g. Squeeze/Unsqueeze axes attr → input conversion)
- Control-flow op support (If, Loop, Scan)
- Dynamic shape / symbolic shape propagation
- Plugin system for 3rd-party op domains
- ONNX model surgery utilities (graph editing before conversion)
- CLI tool (`protofx convert model.onnx`)
- PyPI release workflow
