# ProtoFX

> ONNX → PyTorch `torch.fx.GraphModule` Converter

ProtoFX converts [ONNX](https://onnx.ai/) models into PyTorch [`torch.fx.GraphModule`](https://pytorch.org/docs/stable/fx.html) objects.
Unlike simple weight-loading approaches, ProtoFX preserves the **full graph structure** so that downstream passes—`torch.compile`, quantization, pruning, and custom optimizations—work out of the box.

## Key Features

- **Faithful graph translation** — ONNX ops map to composable `torch.fx` nodes, not opaque forward calls
- **Extensible op registry** — add or override handlers with a single `@register_op` decorator
- **Downstream-ready** — output `GraphModule` integrates seamlessly with `torch.compile`, FX passes, and ONNX-Runtime export round-trips

## Requirements

- Python ≥ 3.12
- PyTorch ≥ 2.0
- ONNX

## Installation

```bash
pip install protofx            # from PyPI (once published)
pip install -e ".[dev]"        # editable install for development
```

## Quick Start

```python
import onnx
from protofx import to_fx

onnx_model = onnx.load("model.onnx")
graph_module = to_fx(onnx_model)

# Now use it like any torch.fx.GraphModule
graph_module.graph.print_tabular()
output = graph_module(input_tensor)
```

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
from protofx.ops import register_op

@register_op("Relu")
def relu_handler(ctx, node):
    x = ctx.get_input(node, 0)
    return ctx.call_function(torch.relu, args=(x,))
```

Each handler lives in `src/protofx/ops/` and is tested with a minimal ONNX fixture in `tests/ops/`.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for planned milestones and feature status.

## License

[Apache License 2.0](LICENSE)
