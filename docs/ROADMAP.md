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
| 📋 | Thin normalized IR (`ir.Graph`, `ir.Node`, `ir.Value`, `ir.TensorType`) |
| 📋 | Refactor current frozen `Node` / `Value` bootstrap into graph-owned mutable IR |
| 📋 | Establish ADR-centered documentation and specification structure |
| 📋 | ONNX importer (`onnx.ModelProto` → IR) |
| 📋 | IR validation and normalization boundary (importer returns `graph.validate()`-clean IR) |
| 📋 | `torch.fx` emitter (IR → `GraphModule`) |
| 📋 | Op handler registry (`@register_op` decorator) |
| 📋 | Basic test infrastructure (`tests/ops/` fixtures) |
| 📋 | CI setup (lint, test) |

---

## Milestone 2 — Core Op Coverage

Handlers for the most common ONNX ops to support real-world models.

| Status | Item |
|--------|------|
| 📋 | Element-wise ops (Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, ...) |
| 📋 | Tensor ops (Reshape, Transpose, Flatten, Squeeze, Unsqueeze, ...) |
| 📋 | Reduction ops (ReduceMean, ReduceSum, ReduceMax, ...) |
| 📋 | Linear algebra (MatMul, Gemm) |
| 📋 | Convolution (Conv, ConvTranspose) |
| 📋 | Normalization (BatchNormalization, LayerNormalization) |
| 📋 | Pooling (MaxPool, AveragePool, GlobalAveragePool) |
| 📋 | Activation (Softmax, LogSoftmax, Gelu, ...) |

---

## Milestone 3 — Model Validation

End-to-end correctness verification against reference ONNX models.

| Status | Item |
|--------|------|
| 📋 | Numerical parity tests vs. ONNX Runtime |
| 📋 | Support for standard model families (ResNet, BERT, ViT, ...) |
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

## Under Consideration 💡

Ideas not yet scheduled. Discuss with `@Architect` before moving to a milestone.

- Control-flow op support (If, Loop, Scan)
- Dynamic shape / symbolic shape propagation
- Plugin system for 3rd-party op domains
- ONNX model surgery utilities (graph editing before conversion)
- CLI tool (`protofx convert model.onnx`)
- PyPI release workflow
