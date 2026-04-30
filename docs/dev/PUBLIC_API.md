---
schema_version: 1
doc_kind: dev
title: Public API
summary: Normative reference for ProtoFX public Python entry points, exported IR symbols, and op registry APIs.
authority: authoritative
keywords: [api, public-api, importers, emitters, ir, ops]
source_of_truth:
  - src/protofx/
  - src/protofx/importers/__init__.py
  - src/protofx/emitters/__init__.py
  - src/protofx/ir/__init__.py
  - src/protofx/ops/__init__.py
related_docs:
  - docs/dev/ARCHITECTURE.md
  - docs/dev/IR.md
  - docs/dev/ir/contracts.md
---

# Public API

<!-- section:purpose -->
## Purpose

This document defines the current public Python API surface that repository documentation may reference directly.

<!-- section:scope -->
## Scope

This page covers exported entry points and module-level public symbols for:

- `protofx.importers`
- `protofx.emitters`
- `protofx.ir`
- `protofx.ops`

It does not restate the full architectural rationale or the full field-by-field IR specification.

<!-- section:contract -->
## Contract

ProtoFX currently exposes its conversion pipeline through submodule entry points rather than a top-level convenience
wrapper.

### Conversion entry points

| Module | Public API | Description |
|--------|------------|-------------|
| `protofx.importers` | `import_model(model_proto: onnx.ModelProto) -> protofx.ir.Graph` | Convert an in-memory ONNX model into validated normalized IR. |
| `protofx.emitters` | `emit_graph(graph: protofx.ir.Graph) -> torch.fx.GraphModule` | Convert validated normalized IR into an executable `torch.fx.GraphModule`. |

The top-level `protofx` package does not currently export a convenience wrapper such as `to_fx()`.

### IR exported surface

`protofx.ir` exports the graph-owned normalized intermediate representation used between ONNX import and FX
emission.

| Symbol | Kind | Purpose |
|--------|------|---------|
| `Graph` | Class | Structural owner of nodes, values, graph inputs, graph outputs, and initializers |
| `Node` | Class | Normalized operation record with graph-managed input and output relationships |
| `Value` | Class | Primary data-flow object for inputs, node outputs, constants, initializers, and sentinels |
| `ValueKind` | Enum | Origin classification for `Value` instances |
| `TensorType` | Dataclass | Immutable tensor metadata attached to each `Value` |
| `DType` | Enum | Backend-neutral element dtype enumeration |
| `Shape` | Type alias | Tensor shape representation |
| `Dim` | Type alias | Single dimension representation |
| `AttributeValue` | Type alias | Normalized Python-native attribute value space for IR node attributes |

### Op registry API

`protofx.ops` exports the process-local op handler registry used by `emit_graph()`.

| API | Purpose |
|-----|---------|
| `register_op(op_type: str, *, opset_range: tuple[int, int] | None = None)` | Register a built-in or extension ONNX op handler for a single op type |
| `dispatch_op(op_type: str, opset_version: int | None = None)` | Resolve the handler for an ONNX op type, optionally enforcing an opset range |
| `list_registry() -> dict[str, tuple[int, int] | None]` | Return a snapshot of the currently registered op handlers |

Importing `protofx.ops` also imports the built-in handler modules so their decorators run at module import time.

<!-- section:invariants -->
## Invariants

- `import_model()` accepts an in-memory `onnx.ModelProto` and returns validated normalized IR.
- `emit_graph()` consumes normalized IR rather than raw ONNX protobuf structures.
- `Graph` remains the structural owner of IR topology and mutation consistency.
- `Node.inputs`, `Node.outputs`, `Value.producer`, and `Value.users` remain observation APIs rather than direct
  structural mutation hooks.
- Op registry duplicate registration is treated as a programmer error.

<!-- section:failure-semantics -->
## Failure Semantics

- `import_model()` raises `ValueError`, `KeyError`, or `NotImplementedError` for invalid or unsupported importer
  cases.
- `emit_graph()` raises `ValueError` for invalid emission state and `NotImplementedError` for missing lowering
  support.
- `dispatch_op()` raises `NotImplementedError` when an op is missing or out of supported opset range.
- `register_op()` raises `ValueError` on duplicate registration.

<!-- section:non-goals -->
## Non-Goals

- This page does not replace `docs/dev/IR.md` or the focused `docs/dev/ir/` specifications.
- This page does not declare exhaustive operator support; use `docs/status/OPSET_COMPATIBILITY.md` for the
  generated registry snapshot.
- This page does not turn every internal helper under `src/protofx/` into public API.

<!-- section:references -->
## References

- Related specs: `docs/dev/IR.md`, `docs/dev/ir/contracts.md`
- Related status docs: `docs/status/OPSET_COMPATIBILITY.md`, `docs/status/SUPPORT_MATRIX.md`
- Source modules: `src/protofx/importers/__init__.py`, `src/protofx/emitters/__init__.py`, `src/protofx/ir/__init__.py`, `src/protofx/ops/__init__.py`
