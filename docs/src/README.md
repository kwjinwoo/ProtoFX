# API Reference

Public API documentation for ProtoFX.

## Modules

| Module | Description |
|--------|-------------|
| `protofx` | Top-level package — main entry points such as `to_fx()` |
| `protofx.ops` | ONNX op handler registry and `@register_op` decorator |
| `protofx.ir` | Internal IR types (Graph, Node, Value, TensorType) |
| `protofx.importers` | ONNX ModelProto → IR conversion |
| `protofx.emitters` | IR → torch.fx.GraphModule conversion |
| `protofx.utils` | Shared utilities (shape inference, type mapping) |

## IR Notes

ProtoFX IR is a graph-owned normalized representation.

- `Graph` owns topology, membership, and use-def consistency.
- `Node` and `Value` remain convenient public objects for importer, validator, and emitter code.
- `TensorType` remains the immutable metadata object attached to values.
