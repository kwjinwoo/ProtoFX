# API Reference

Public API documentation for ProtoFX.

## Conversion Entry Points

ProtoFX currently exposes its conversion pipeline through submodule entry points.

| Module | Public API | Description |
|--------|------------|-------------|
| `protofx.importers` | [`import_model()`](importers.md) | Convert `onnx.ModelProto` into a validated normalized IR graph |
| `protofx.emitters` | [`emit_graph()`](emitters.md) | Convert normalized IR into `torch.fx.GraphModule` |

The top-level `protofx` package does not currently provide a convenience wrapper such as `to_fx()`.

## Modules

| Module | Description |
|--------|-------------|
| `protofx` | Top-level package namespace and package metadata. No convenience `to_fx()` wrapper is currently exported. |
| [`protofx.ops`](ops.md) | ONNX op handler registry, dispatch helpers, and `@register_op` decorator |
| [`protofx.ir`](ir.md) | Public IR surface (`Graph`, `Node`, `Value`, `TensorType`, and related types) |
| [`protofx.importers`](importers.md) | ONNX ModelProto → IR conversion |
| [`protofx.emitters`](emitters.md) | IR → torch.fx.GraphModule conversion |
| `protofx.utils` | Shared utilities (shape inference, type mapping) |

## IR Notes

ProtoFX IR is a graph-owned normalized representation.

- `Graph` owns topology, membership, and use-def consistency.
- `Node` and `Value` remain convenient public objects for importer, validator, and emitter code.
- `TensorType` remains the immutable metadata object attached to values.

For architectural rationale, see `docs/adr/0001-thin-graph-owned-ir.md`.
For detailed IR specifications, see `docs/dev/IR.md` and the `docs/dev/ir/` documents.
