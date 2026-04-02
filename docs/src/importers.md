# Importers API

Public API reference for `protofx.importers`.

## Module Contract

`protofx.importers` is the ONNX-facing boundary of ProtoFX. It accepts an in-memory `onnx.ModelProto` and
returns normalized IR that is ready for validation-sensitive consumers such as the emitter.

For the authoritative pipeline contract, see [../dev/ir/contracts.md](../dev/ir/contracts.md).

## Public API

### `import_model(model_proto: onnx.ModelProto) -> protofx.ir.Graph`

Convert an ONNX model into a normalized `protofx.ir.Graph`.

#### Parameters

| Name | Type | Description |
|------|------|-------------|
| `model_proto` | `onnx.ModelProto` | ONNX model already loaded in memory. Load from disk with `onnx.load()` or build with `onnx.helper` before calling `import_model()`. |

#### Returns

A validated `protofx.ir.Graph`.

The returned graph preserves the public Milestone 1 importer boundary:

- graph inputs and graph outputs keep ONNX order
- initializers are stored as `ValueKind.INITIALIZER` values with `numpy.ndarray` payloads
- ONNX `Constant` ops are inlined as `ValueKind.CONSTANT` values instead of being kept as explicit IR nodes
- provenance retained for Milestone 1 includes `Graph.name`, `Node.name`, `Node.op_type`, `Node.domain`, `Node.opset_version`, and `Value.name` when available

#### Import Behavior

- Runs `onnx.shape_inference.infer_shapes()` opportunistically before import. If shape inference fails, import still proceeds and missing dtype or shape metadata remains explicit as `None`.
- Deduplicates initializer names that also appear in `graph.input`, matching the ONNX opset `< 9` compatibility pattern without violating IR input/initializer separation.
- Normalizes supported ONNX attributes into Python-native values: `int`, `float`, `bytes`, `list[int]`, `list[float]`, and `list[bytes]`.
- Normalizes `Conv` and `ConvTranspose` `auto_pad` values into explicit `pads` when enough static metadata is available.
- Calls `graph.validate()` before returning, so invalid IR must fail at the importer boundary instead of reaching emission.

#### Raises

| Exception | When |
|-----------|------|
| `ValueError` | A required `Constant` tensor payload is missing, or the produced graph violates IR invariants during `graph.validate()`. |
| `KeyError` | A node input name cannot be resolved from prior inputs, initializers, or previously imported outputs. |
| `NotImplementedError` | The model uses an unsupported ONNX attribute type, or `auto_pad` normalization requires metadata that is unavailable. |

#### Example

```python
import onnx

from protofx.importers import import_model

model = onnx.load("model.onnx")
graph = import_model(model)

print(graph.name)
print(len(graph.inputs), len(graph.initializers), len(graph.nodes), len(graph.outputs))
```

## Notes for Callers

- `import_model()` expects an in-memory `onnx.ModelProto`; it does not load paths or raw bytes for you.
- The returned graph is validated, but consumers that need dependency-safe traversal should still use `graph.topological_sort()` rather than assuming `graph.nodes` is the authoritative ordering API.
