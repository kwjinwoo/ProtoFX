# IR Type System

This document records the value and tensor metadata model used by ProtoFX IR.

## Design Summary

- `TensorType` is the immutable metadata object attached to `Value`.
- `Value` is the primary data-flow unit.
- `ValueKind` classifies the origin and role of a value.
- Type mapping utilities sit at the importer and emitter boundaries rather than inside the IR core.

## TensorType

`ir.TensorType` carries lightweight tensor metadata.

The accepted model is:

- `dtype: DType | None`
- `shape: Shape`

`TensorType` remains immutable so metadata replacement is explicit and safe.

## DType

`ir.DType` is a backend-neutral enum.

Its integer values mirror `onnx.TensorProto.DataType` so importer conversion can stay simple without forcing
the IR layer to depend on `onnx`.

## Dim and Shape

- `Dim = int | str | None`
- `Shape = tuple[Dim, ...] | None`

This representation covers concrete, symbolic, and unknown dimensions while keeping the Milestone 1 type model
lightweight.

## Value and ValueKind

`ir.Value` represents one data-flow object in a graph.

Expected field shape:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Stable internal identifier |
| `kind` | `ValueKind` | Origin classification |
| `tensor_type` | `TensorType` | Tensor metadata |
| `name` | `str | None` | Original ONNX name for source provenance |
| `producer` | `Node | None` | Producing node, graph-managed |
| `users` | `list[tuple[Node, int]]` | Consumer and slot pairs, graph-managed |

`ValueKind` should distinguish at least:

- `GRAPH_INPUT`
- `NODE_OUTPUT`
- `SENTINEL`
- `CONSTANT`
- `INITIALIZER`

## Boundary Utilities

Two boundary utilities bridge the IR type model to external systems:

| Function | Direction | Dependency |
|----------|-----------|------------|
| `onnx_dtype_to_ir(elem_type)` | ONNX → IR | `onnx` |
| `ir_dtype_to_torch(dtype)` | IR → PyTorch | `torch` |

`torch` imports should remain lazy on the emitter side.
