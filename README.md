# ProtoFX

> ONNX → PyTorch `torch.fx.GraphModule` Converter

ProtoFX converts [ONNX](https://onnx.ai/) models into PyTorch [`torch.fx.GraphModule`](https://pytorch.org/docs/stable/fx.html) objects.
Unlike simple weight-loading approaches, ProtoFX preserves the **full graph structure** so that downstream
passes can consume the resulting `GraphModule` directly.

## Key Features

- **Faithful graph translation** — ONNX ops map to composable `torch.fx` nodes, not opaque forward calls
- **Extensible op registry** — add or override handlers with a single `@register_op` decorator
- **Downstream-oriented** — output `GraphModule` is structurally compatible with `torch.compile`, FX passes, and quantization workflows (downstream integration is under active validation — see [Roadmap](docs/ROADMAP.md))

## Requirements

- Python ≥ 3.12
- PyTorch ≥ 2.5
- ONNX ≥ 1.16

## Installation

```bash
pip install protofx            # from PyPI (once published)
pip install -e ".[dev]"        # editable install for development
```

## Quick Start

ProtoFX exposes a two-step pipeline — **import** then **emit**:

```python
import onnx

from protofx.emitters import emit_graph
from protofx.importers import import_model

# 1. Import ONNX model into normalized IR
onnx_model = onnx.load("model.onnx")
ir_graph = import_model(onnx_model)

# 2. Emit torch.fx.GraphModule from IR
graph_module = emit_graph(ir_graph)

# Use it like any torch.fx.GraphModule
graph_module.graph.print_tabular()
(output,) = graph_module(input_tensor)
```

> **Note:** The top-level `protofx` package does not currently export a convenience `to_fx()` wrapper.
> Use `import_model()` and `emit_graph()` directly.

## Architecture

```
src/protofx/
├── importers/       # ONNX graph → IR (intermediate representation)
├── ir/              # Internal graph IR nodes and types
├── emitters/        # IR → torch.fx Graph construction
├── ops/             # Per-ONNX-op conversion handlers (one file per domain)
└── utils/           # Shared helpers (shape inference, type mapping)
```

| Layer | Responsibility |
|-------|----------------|
| **Importers** | Read `onnx.ModelProto` and build an internal IR |
| **IR** | Lightweight graph representation decoupled from ONNX and FX |
| **Emitters** | Walk the IR and emit `torch.fx` nodes via `torch.fx.Graph` |
| **Ops** | Per-op conversion handlers registered by ONNX op name/domain |
| **Utils** | Shape inference, type mapping, and shared helpers |

## Current Coverage

| Family | Validated Models | Model-Level Downstream Validation |
|--------|-----------------|-----------------------------------|
| Smoke / baseline | SqueezeNet | `torch.compile`, PT2E quantization |
| Vision | ResNet18, ViT-B/16 | `torch.compile` on ResNet18 |
| NLP | BERT | `torch.compile` on BERT |

For the full model-by-task matrix and status definitions, see [docs/dev/SUPPORT_MATRIX.md](docs/dev/SUPPORT_MATRIX.md).

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Lint & format
ruff check src/ tests/
ruff format src/ tests/

# Run tests
pytest tests/ -v
```

### Adding an Op Handler

```python
import torch

from protofx.ops import register_op


@register_op("Relu", opset_range=(11, 21))
def relu_handler(node, args, fx_graph, module):
    return [fx_graph.call_function(torch.relu, args=(args[0],))]
```

Each handler receives the IR `node`, a list of resolved FX `args`, the `fx_graph` under construction, and the
root `module`. It returns a list of `torch.fx.Node` — one per IR output.

Handlers live in `src/protofx/ops/` and are tested with minimal ONNX fixtures in `tests/ops/`.
See [docs/src/ops.md](docs/src/ops.md) for the full registry API reference.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for planned milestones and feature status.

## License

[Apache License 2.0](LICENSE)
