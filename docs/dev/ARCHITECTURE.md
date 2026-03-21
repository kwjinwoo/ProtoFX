# Architecture

## Overview

ProtoFX is a three-stage pipeline that converts ONNX models into PyTorch `torch.fx.GraphModule` objects.

```
ONNX ModelProto ──▶ Importer ──▶ Thin Normalized IR ──▶ Validation / Analysis ──▶ Emitter ──▶ torch.fx.GraphModule
```

## Directory Structure

```
src/protofx/
├── importers/       # ONNX graph → IR (intermediate representation)
├── ir/              # Internal graph IR nodes and types
├── emitters/        # IR → torch.fx Graph construction
├── ops/             # Per-ONNX-op conversion handlers (one file per domain)
└── utils/           # Shared helpers (shape inference, type mapping)
```

## Components

### Importers

Reads `onnx.ModelProto` and converts it into the internal IR.

- Parses nodes, tensors, and initializer values from the ONNX graph
- Handles branching based on ONNX opset version
- Recursively imports subgraphs (control flow)

### IR (Intermediate Representation)

A thin normalized graph representation independent of both ONNX and `torch.fx`.

- Holds normalized node, value, constant, and tensor type information
- Acts as the semantic boundary between Importer and Emitter
- Provides a stable target for validation and analysis before backend emission
- See [IR.md](IR.md) for details

### Emitters

Traverses the IR and creates nodes in `torch.fx.Graph`.

- Constructs `torch.fx.Graph` and `torch.fx.GraphModule`
- Converts each IR node to an FX node via the op handler registry
- Consumes normalized IR data rather than raw ONNX protobuf structures
- Keeps `torch` imports lazy to optimize import speed

### Ops

Contains per-ONNX-op conversion handlers.

- Registered with the `@register_op("opname")` decorator
- Organized by domain (e.g., `nn.py`, `math.py`)
- Each handler returns `torch.fx.Node`(s)

### Utils

Shared helper modules.

- Shape inference utilities
- ONNX ↔ PyTorch type mapping
- Tensor conversion helpers

## Architectural Boundary

ProtoFX intentionally separates three concerns:

1. **Importer**: ONNX-aware parsing and normalization.
2. **IR and validation**: internal graph structure, metadata, and structural checks.
3. **Emitter**: FX-aware lowering from normalized IR.

This boundary is important for two reasons:

- ONNX protobuf details, opset quirks, and attribute decoding should not leak into FX emission code.
- FX-specific lowering decisions should not distort the imported graph model.

The project does **not** treat IR as a full compiler framework. It is a minimal normalization layer chosen to
support downstream compatibility, testing, and future expansion without over-design.
