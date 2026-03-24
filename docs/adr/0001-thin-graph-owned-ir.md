# ADR-0001: Thin, Graph-Owned Normalized IR

- Status: Accepted
- Date: 2026-03-22

## Context

ProtoFX converts `onnx.ModelProto` objects into `torch.fx.GraphModule` objects. A direct ONNX-to-FX pipeline
looks smaller at first, but it pushes parsing, normalization, validation, and backend lowering into a single
stage.

That coupling does not fit the project roadmap. ProtoFX needs a boundary that supports importer evolution,
structural validation, downstream FX compatibility, and future expansion such as control flow and symbolic
shape work.

## Decision

ProtoFX uses a thin normalized IR between ONNX import and `torch.fx` emission.

The accepted IR shape is graph-owned and transform-friendly:

- `ir.Graph` owns topology, membership, and use-def consistency.
- `ir.Node` and `ir.Value` remain mutable IR entities managed through graph-aware APIs.
- `ir.TensorType` remains an immutable value object.
- Public convenience accessors such as `value.producer`, `value.users`, `node.inputs`, and `node.outputs`
  remain available, but structural consistency is maintained by `ir.Graph`.

## Consequences

### Benefits

- Importer logic stays separate from FX lowering details.
- Validation and analysis can target a stable normalized representation.
- Testing boundaries become cleaner: importer, IR validation, and emitter can be tested independently.
- Future work such as control flow and alternate backends remains possible without redesigning the whole
  pipeline.

### Costs

- ProtoFX must define and maintain IR invariants explicitly.
- The project takes on a small but real specification burden for graph ownership, value kinds, and mutation
  APIs.
- Early implementation is slightly slower than a prototype that lowers ONNX directly into FX.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| Direct ONNX proto to FX conversion | Too much coupling between parsing, normalization, and emission |
| Full compiler-style IR | Too heavy for ProtoFX's current scope and roadmap |
| Frozen dataclass `Node` / `Value` model | Fights graph normalization and mutation requirements |

## Derived Specifications

- `docs/dev/IR.md`
- `docs/dev/ir/invariants.md`
- `docs/dev/ir/type-system.md`
- `docs/dev/ir/graph-model.md`
- `docs/dev/ir/contracts.md`
