# Architecture

## Overview

ProtoFX is a three-stage pipeline that converts ONNX models into PyTorch `torch.fx.GraphModule` objects.

```
ONNX ModelProto ──▶ Importer ──▶ IR ──▶ Emitter ──▶ torch.fx.GraphModule
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

A lightweight graph representation independent of both ONNX and `torch.fx`.

- Holds node, edge, and tensor type information
- Acts as a decoupling layer between Importer and Emitter
- See [IR.md](IR.md) for details

### Emitters

Traverses the IR and creates nodes in `torch.fx.Graph`.

- Constructs `torch.fx.Graph` and `torch.fx.GraphModule`
- Converts each IR node to an FX node via the op handler registry
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
