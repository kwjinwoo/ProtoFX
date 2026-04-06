# Ops API

Public API reference for `protofx.ops`.

## Module Contract

`protofx.ops` is the operator registry used by `protofx.emitters.emit_graph()` to lower normalized IR nodes into
`torch.fx` nodes.

Importing `protofx.ops` also imports the built-in handler modules so their `@register_op(...)` decorators run at
module import time. Registry state is therefore process-local and reflects whichever handlers have been imported
into the current Python process.

For the emitter-side pipeline contract, see [../dev/ir/contracts.md](../dev/ir/contracts.md).

## Handler Signature

Registered handlers must match the following callable shape:

```python
def handler(
    node: protofx.ir.Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    ...
```

- `node` is the IR node being lowered.
- `args` is the ordered list of resolved FX inputs. Omitted optional ONNX inputs are represented as `None`.
- `fx_graph` is the graph under construction.
- `module` is the root module used for buffer or parameter registration.
- The return value must contain exactly one `torch.fx.Node` per IR output.

## Public API

### `register_op(op_type: str, *, opset_range: tuple[int, int] | None = None)`

Register an ONNX op handler for a single `op_type`.

#### Parameters

| Name | Type | Description |
|------|------|-------------|
| `op_type` | `str` | ONNX operator type such as `"Relu"` or `"Conv"`. |
| `opset_range` | `tuple[int, int] | None` | Inclusive supported opset range. `None` means the handler does not declare version constraints. |

#### Behavior

- Registration is one handler per `op_type`, not one handler per version range.
- Re-registering an already registered `op_type` raises `ValueError` immediately.
- The decorator returns the original handler unchanged after registration.

### `dispatch_op(op_type: str, opset_version: int | None = None) -> OpHandler`

Look up the handler for an ONNX op type.

#### Parameters

| Name | Type | Description |
|------|------|-------------|
| `op_type` | `str` | ONNX operator type to resolve. |
| `opset_version` | `int | None` | Model opset used for compatibility checking. `None` skips version enforcement. |

#### Behavior

- If `op_type` is not registered, `dispatch_op()` raises `NotImplementedError`.
- If a handler declares `opset_range`, the range is inclusive on both ends.
- If `opset_version` falls outside that inclusive range, `dispatch_op()` raises `NotImplementedError`.
- If `opset_version` is `None`, version validation is skipped.
- If the handler was registered without `opset_range`, any `opset_version` value is accepted.

### `list_registry() -> dict[str, tuple[int, int] | None]`

Return a snapshot of the current registry state.

The returned dict maps each registered ONNX op type to either:

- an inclusive `(min_opset, max_opset)` tuple
- `None` when the handler declared no version constraint

This API is used by the generated compatibility matrix in `docs/dev/OPSET_COMPATIBILITY.md`.

## Duplicate Registration and Version Semantics

The registry behavior is intentionally strict.

- Duplicate registration is a programmer error and fails with `ValueError`.
- Missing handlers and out-of-range opset dispatch are treated as unsupported lowering paths and fail with
  `NotImplementedError`.
- Range boundaries are inclusive, so `(11, 21)` supports both opset 11 and opset 21.

## Example

```python
import torch

from protofx.ops import register_op


@register_op("Relu", opset_range=(11, 21))
def relu_handler(node, args, fx_graph, module):
    return [fx_graph.call_function(torch.relu, args=(args[0],))]
```

For emitter integration details, see [emitters.md](emitters.md).
