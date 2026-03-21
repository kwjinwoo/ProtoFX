# IR (Intermediate Representation)

This document records the architectural decision for the ProtoFX intermediate representation.

## Decision

ProtoFX will use a **thin normalized IR** between ONNX import and `torch.fx` emission.

The IR is intentionally small. It is not a second copy of ONNX and it is not a general-purpose compiler IR.
Its purpose is to normalize ONNX graph semantics into a backend-neutral form that is simple to validate,
analyze, and emit into `torch.fx.GraphModule`.

## Status

Accepted.

## Context

ProtoFX converts `onnx.ModelProto` objects into `torch.fx.GraphModule` objects. A direct conversion pipeline
from ONNX proto structures to FX nodes is possible, but it pushes too many responsibilities into one stage:

- parsing ONNX serialization details
- resolving opset and domain-specific differences
- normalizing attributes and constants
- tracking value provenance and multi-output nodes
- validating shape and dtype assumptions
- selecting the correct FX emission form

That coupling is manageable for a small proof of concept, but it does not scale well to the project goals in
[../ROADMAP.md](../ROADMAP.md): broader op coverage, validation, `torch.compile` compatibility, FX-based
quantization, and eventual control-flow support.

## Why Not Convert ONNX Proto Directly to FX

Direct conversion has one real advantage: a smaller initial implementation surface.

That benefit is outweighed by structural problems:

- **Importer/emitter coupling**: ONNX parsing details leak into backend emission logic.
- **Version branching in the wrong place**: opset differences end up inside FX-specific code.
- **Poor reuse**: graph validation, diagnostics, and non-FX backends become harder to add.
- **Weaker testing boundaries**: only end-to-end testing remains practical; importer and emitter cannot be
	validated independently.
- **Control-flow complexity**: `If`, `Loop`, and `Scan` require explicit handling of nested regions and captured
	values. That is difficult to represent cleanly without an intermediate graph model.
- **Inconsistent constant handling**: initializers, `Constant` nodes, scalars, and omitted optional inputs tend
	to be normalized ad hoc inside op handlers.

The direct path is reasonable for a narrow demo converter. It is not the right foundation for ProtoFX.

## Design Goals

- **Normalization first**: convert ONNX serialization details into a stable internal semantic form.
- **Thin representation**: keep only the information needed for validation and emission.
- **Backend neutrality**: importer logic must not depend on `torch.fx`, and emitter logic must not depend on
	raw ONNX proto APIs.
- **Explicit unknowns**: unknown shape, dtype, optional inputs, and symbolic dimensions must be representable.
- **Good testability**: importer, IR validation, and emitter should be testable as separate units.
- **Future headroom**: the model should support control flow, symbolic shapes, and alternate emitters without
	redesigning the entire pipeline.

## Non-Goals

- Reproducing the ONNX protobuf schema one-to-one.
- Introducing a full compiler optimization IR.
- Encoding every backend-specific lowering detail in the IR.
- Performing aggressive graph rewrites in Milestone 1.

## IR Scope

The IR is the boundary between three concerns:

1. **Import**: read ONNX protobuf structures and normalize them.
2. **Validate / analyze**: check structural and type assumptions before backend emission.
3. **Emit**: lower the normalized graph into `torch.fx`.

The intended pipeline is:

```
onnx.ModelProto
		-> importer
		-> ir.Graph
		-> validation / analysis passes
		-> emitter
		-> torch.fx.GraphModule
```

## Core Model

ProtoFX IR is **value-centric**.

The important abstraction is not an explicit `Edge` object but the named value flowing between producers and
consumers. Logical edges exist, but they are represented by value references.

### Graph

`ir.Graph` represents one normalized graph region.

It should contain:

- graph inputs in a stable order
- graph outputs in a stable order
- nodes in topological order
- a value table for looking up producers, users, and metadata
- constant and initializer bindings
- graph-level metadata such as opset imports and source provenance

The graph must be simple to traverse without access to ONNX proto internals.

### Value

`ir.Value` is the central data-flow object.

It should contain:

- a stable identifier or name
- a reference to its producer node, if any
- a list of user nodes
- tensor metadata when known
- flags or kind information for graph inputs, constants, and optional values
- room for symbolic shape information and future annotations

Using `Value` instead of a heavy `Edge` model simplifies several hard cases:

- multi-output operators
- optional or omitted inputs
- values captured by nested subgraphs
- constants lifted from initializers or `Constant` nodes

### Node

`ir.Node` represents one normalized operation.

It should contain:

- `op_type`
- `domain`
- normalized attributes in Python-native form
- ordered input `Value` references
- ordered output `Value` references
- source metadata for diagnostics, such as the original ONNX node name

Important constraint: the node must not expose raw `AttributeProto` parsing concerns to the emitter.

### TensorType

`ir.TensorType` carries lightweight tensor metadata.

It should contain:

- dtype when known
- shape when known
- rank information when partially known
- symbolic dimensions when available
- a way to represent unknown values explicitly

The IR must distinguish between:

- unknown shape
- scalar shape
- empty dimensions
- partially known symbolic shapes

### Constants and Initializers

ProtoFX must normalize all constant-like inputs into a consistent representation before emission.

That includes:

- graph initializers
- `Constant` op outputs
- scalar literals derived from attributes or static inputs
- omitted optional inputs when an op uses a positional placeholder

Without this normalization, handlers tend to grow backend-specific special cases.

## Normalization Rules

The importer is responsible for converting ONNX-specific forms into a single internal representation.

At minimum, import normalization should:

- resolve graph inputs versus initializers
- decode attributes into Python-native values
- assign explicit `Value` objects to all node outputs
- preserve output ordering for multi-output ops
- represent omitted optional inputs explicitly
- retain opset and domain information needed to interpret node semantics
- preserve source names for diagnostics and debugging

The emitter should consume normalized nodes and values, not interpret raw protobuf details.

## Validation and Analysis Boundary

The IR exists partly to create a clean place for validation.

Before FX emission, ProtoFX should be able to check:

- graph well-formedness
- unsupported operator patterns
- mismatched input arity
- invalid attribute combinations
- basic dtype and shape expectations when metadata is available
- region interface consistency for future control-flow ops

This is architecturally important. Failures should occur as early as possible, with errors phrased in terms of
the normalized graph rather than leaking backend internals.

## Relationship to ONNX and FX

The mapping is intentionally asymmetric:

- ONNX is the source serialization and semantic input.
- IR is the normalized internal semantic model.
- FX is one backend representation for executable graphs.

The conversion flow is therefore:

```
onnx.NodeProto  ->  ir.Node   ->  torch.fx.Node
onnx.GraphProto ->  ir.Graph  ->  torch.fx.Graph
```

What changes across the boundary:

- protobuf-specific storage details are removed
- constants and inputs are normalized
- value flow becomes explicit
- backend-specific emission choices are deferred until the emitter

## Consequences

### Benefits

- Cleaner separation between ONNX import and FX emission.
- Better unit-testing boundaries.
- A stable place for validation and diagnostics.
- Easier support for control flow, symbolic shapes, and plugin domains later.
- Less duplication in op handlers.

### Costs

- One extra internal data model must be designed and maintained.
- Early implementation will take longer than a direct converter.
- Documentation must clearly define what belongs in the IR and what does not.

Those costs are acceptable. They buy a more stable architecture for the roadmap ProtoFX already targets.

## Recommended Initial Surface

Milestone 1 should keep the IR small. A good initial scope is:

- `ir.Graph`
- `ir.Node`
- `ir.Value`
- `ir.TensorType`

No additional node taxonomy is required yet. If control flow arrives later, region or subgraph support can be
added incrementally rather than designed upfront in full generality.

## Summary

ProtoFX IR is not an optional abstraction layer added for theoretical purity.

It is the normalization boundary that turns ONNX serialization-oriented structures into a simple internal graph
model suitable for validation and `torch.fx` emission. For ProtoFX, that boundary is worth keeping, but only as
a thin normalized IR.
