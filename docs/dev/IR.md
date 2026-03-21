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

## IR Invariants

This section defines the accepted invariants for the initial ProtoFX IR.

These invariants are intended to be a semi-public developer contract. Contributors may rely on them when
working on import, validation, and emission, but the document still reflects an early-stage architecture and
may be refined in later milestones if the project discovers a better boundary.

### 1. Value-Centric Data Flow

All data-flow units are represented as `Value` objects.

This includes:

- graph inputs
- node outputs
- constants
- initializers
- explicit placeholders for omitted optional inputs

ProtoFX therefore treats values, not edges, as the primary graph connectivity model.

### 2. Constant Interface with Source Preservation

Initializers and `Constant` op outputs share a common constant interface inside the IR.

However, the IR must preserve their source kind as metadata so that later stages can still distinguish:

- graph-sourced initializers
- node-produced constants
- other literal-like constant forms introduced during normalization

This keeps the importer and emitter simple without discarding provenance needed for diagnostics and future
lowering policies.

### 3. Explicit Omitted Optional Inputs

An omitted optional ONNX input is represented by a dedicated sentinel `Value`.

This is required to preserve positional input contracts without overloading Python `None` semantics or silently
removing inputs from ordered input lists.

The sentinel must be distinguishable from ordinary runtime values.

### 4. Multi-Output Nodes Produce Independent Values

A node with multiple outputs must produce one distinct `Value` per output.

Those outputs remain ordered according to the ONNX operator contract.

ProtoFX does not model node outputs as a tuple-like aggregate object in the base IR.

### 5. Stable Internal Identity

`Value` identity is based on stable internal identifiers, not directly on ONNX names.

Original ONNX names should be preserved as source metadata when available, but internal graph correctness must
not depend on them.

This avoids importer fragility caused by unnamed outputs, duplicate names, or backend-specific naming quirks.

### 6. Rich Unknown Representation for Tensor Metadata

`TensorType` must be able to distinguish at least the following cases:

- unknown shape
- scalar shape
- empty dimensions
- partially known symbolic dimensions
- fully known shapes

ProtoFX does not collapse all incomplete metadata into a single generic "unknown" state.

### 7. Tensor Metadata Lives on Values

`TensorType` is attached to `Value`, not to `Node`.

This keeps type information aligned with actual data flow and avoids duplicating metadata across producers and
consumers.

### 8. Single Producer Invariant

Every non-input, non-sentinel `Value` must have exactly one producer.

This establishes an SSA-like discipline that simplifies validation, diagnostics, and future graph reasoning.

Expected exceptions are only those values that are created by graph boundaries or dedicated normalization rules,
such as graph inputs and omitted-input sentinels.

### 9. Ordered Inputs and Outputs

Node inputs and outputs preserve ONNX positional order.

The IR may normalize meaning, but it must not reorder operator interfaces into a custom semantic layout.

This keeps importer, validation, and emitter behavior aligned with ONNX schemas.

### 10. Attribute Normalization Happens in the Importer

All node attributes are normalized into Python-native forms during import.

The emitter must not depend on raw ONNX `AttributeProto` structures or re-interpret protobuf-specific storage
details.

This is one of the main architectural boundaries in ProtoFX.

### 11. Source Provenance Is Preserved

The IR should retain enough source metadata to produce useful diagnostics and debugging output.

At minimum, the IR should preserve, when available:

- original node names
- output names
- operator type and domain
- opset context
- graph boundary provenance

Internal identifiers are authoritative for correctness, but source metadata remains important for usability.

### 12. Graph Nodes Remain in Topological Order

`ir.Graph.nodes` must remain topologically ordered.

Any importer or future pass that would violate this invariant is responsible for restoring valid order before the
graph is handed to downstream stages.

This makes traversal and emission predictable by construction.

### 13. Graph Inputs and Initializers Remain Distinct

Graph inputs and initializers are semantically distinct and must remain distinct in the IR.

Normalization may unify how their values are accessed, but the graph boundary contract must preserve which
values are runtime inputs and which are statically bound constants.

### 14. Control-Flow Readiness Without Early Over-Commitment

Milestone 1 invariants should preserve future extensibility for control flow, but they do not yet define a full
region or subgraph ownership model.

ProtoFX should avoid baking in assumptions that would prevent `If`, `Loop`, or `Scan`, while also avoiding a
premature full control-flow abstraction before those operators are designed in detail.

### 15. Strict Validation Policy

IR validation should be strict.

When the importer produces invalid IR, ProtoFX should fail as early as possible rather than defer structural or
semantic problems to the emitter.

Validation is expected to enforce:

- graph well-formedness
- producer and user consistency
- ordered interface consistency
- required attribute presence and normalized form
- shape and dtype constraints when enough metadata is available

Strict validation does not imply pretending unknown metadata is known. Unknowns remain explicit and valid when
the source model does not provide enough information.

### 16. Invariants Before Field-Level API

This document fixes invariants first and intentionally stops short of standardizing a complete field-by-field API
for each IR type.

Concrete field lists and helper methods may evolve later as implementation begins, but those details must remain
consistent with the invariants defined here.

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

## Recommended Development Order

IR elements should be developed in dependency order rather than by container size or class name.

The recommended sequence is:

1. **IR invariants**
2. **`TensorType`**
3. **`Value`**
4. **`Node`**
5. **`Graph`**
6. **Constant and initializer normalization**
7. **Validation and analysis boundary**
8. **Importer-to-IR contract**
9. **IR-to-emitter contract**

### 1. IR invariants

Before implementing concrete classes, define the structural rules that all IR objects must satisfy.

Examples include:

- every data-flow result is represented as a `Value`
- optional or omitted inputs have an explicit representation
- node inputs and outputs are ordered
- initializers and constants follow one normalization policy
- unknown shape and scalar shape are not conflated

Without these invariants, later class design tends to drift and requires rework.

### 2. `TensorType`

`TensorType` should be defined early because it is a leaf dependency for both `Value` and validation.

It should establish:

- dtype representation
- known versus unknown rank
- known versus symbolic dimensions
- scalar and empty-shape handling

### 3. `Value`

`Value` comes before `Node` because ProtoFX IR is value-centric.

This type defines the graph's actual data-flow model:

- producer reference
- user list
- metadata attachment
- value kind such as input, constant, or optional placeholder

### 4. `Node`

Only after `Value` is stable should `Node` be defined.

`Node` should remain a normalized semantic operation with:

- `op_type`
- `domain`
- normalized attributes
- ordered input values
- ordered output values
- source metadata for diagnostics

### 5. `Graph`

`Graph` should be assembled after `Value` and `Node` semantics are clear.

At that point it can own:

- ordered graph inputs and outputs
- topologically ordered nodes
- value lookup tables
- graph-level metadata and constant bindings

Defining `Graph` too early usually produces a shallow container without meaningful invariants.

### 6. Constant and initializer normalization

This should be treated as a dedicated design step rather than a side effect of node import.

The importer must decide how to represent:

- graph initializers
- `Constant` op outputs
- scalar literals
- omitted optional inputs

This is one of the main points where direct conversion designs become inconsistent.

### 7. Validation and analysis boundary

Validation is not an optional add-on. It is part of the reason the IR exists.

Once the core model is defined, ProtoFX should add checks for:

- graph well-formedness
- producer and user consistency
- arity mismatches
- invalid attribute combinations
- basic shape and dtype sanity when metadata is available

### 8. Importer-to-IR contract

After the IR model is stable, define exactly what the importer guarantees.

That contract should cover:

- attribute normalization
- input versus initializer resolution
- output naming or identification policy
- opset and domain metadata attachment
- source provenance retention

### 9. IR-to-emitter contract

The emitter contract should be finalized last.

Its purpose is to ensure that backend emission consumes only normalized IR data and does not rely on raw ONNX
protobuf details.

### What Not to Prioritize Early

The following should stay out of the initial development sequence unless a concrete requirement appears:

- an explicit `Edge` class
- a generic pass manager
- optimization rewrite infrastructure
- fully general control-flow region modeling
- backend-agnostic compiler-style abstractions

ProtoFX should first establish a stable thin normalized IR, not a full compiler framework.

## Summary

ProtoFX IR is not an optional abstraction layer added for theoretical purity.

It is the normalization boundary that turns ONNX serialization-oriented structures into a simple internal graph
model suitable for validation and `torch.fx` emission. For ProtoFX, that boundary is worth keeping, but only as
a thin normalized IR.
