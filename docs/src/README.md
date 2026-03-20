# API Reference

Public API documentation for ProtoFX.

## Modules

| Module | Description |
|--------|-------------|
| `protofx` | Top-level package — main entry points such as `to_fx()` |
| `protofx.ops` | ONNX op handler registry and `@register_op` decorator |
| `protofx.ir` | Internal IR types (Graph, Node, Edge, TensorType) |
| `protofx.importers` | ONNX ModelProto → IR conversion |
| `protofx.emitters` | IR → torch.fx.GraphModule conversion |
| `protofx.utils` | Shared utilities (shape inference, type mapping) |
