# Roadmap

This document tracks the planned development milestones for ProtoFX.
Decisions that affect architecture are recorded in [docs/dev/ARCHITECTURE.md](dev/ARCHITECTURE.md).

> **Status key**: ✅ Done · 🚧 In Progress · 📋 Planned · 💡 Under Consideration

---

## Milestone 1 — Foundation

Core infrastructure required before any op handler can be written.

Implementation order for IR foundations is documented in [docs/dev/IR.md](dev/IR.md).

The accepted IR direction for this milestone is a graph-owned mutable IR: `ir.Graph` owns structural
consistency, while `ir.Node` and `ir.Value` remain convenient public objects without frozen-dataclass
constraints.

| Status | Item |
|--------|------|
| 📋 | Thin normalized IR (`ir.Graph`, `ir.Node`, `ir.Value`, `ir.TensorType`) |
| 📋 | Refactor current frozen `Node` / `Value` bootstrap into graph-owned mutable IR |
| 📋 | ONNX importer (`onnx.ModelProto` → IR) |
| 📋 | IR validation and normalization boundary |
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

## Milestone 1 Implementation Order

The foundation milestone should be implemented in this order:

1. Finalize IR invariants.
2. Keep `TensorType` as the immutable metadata leaf type.
3. Introduce `ir.Graph` ownership and mutation APIs.
4. Refactor `Value` to graph-managed producer/user relationships.
5. Refactor `Node` to graph-managed creation and ordered interfaces.
6. Implement constant and initializer normalization.
7. Add validation for graph consistency and mutation safety.
8. Build the importer against the graph-owned IR contract.
9. Build the emitter against normalized graph APIs.
