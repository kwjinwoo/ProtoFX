# IR (Intermediate Representation)

This document records the architectural decision for the ProtoFX intermediate representation.

## Decision

ProtoFX will use a **thin normalized IR** between ONNX import and `torch.fx` emission.

The IR is intentionally small. It is not a second copy of ONNX and it is not a general-purpose compiler IR.
Its purpose is to normalize ONNX graph semantics into a backend-neutral form that is simple to validate,
analyze, and emit into `torch.fx.GraphModule`.

The IR will be **graph-owned and transform-friendly**:

- `ir.Graph` is the owner of topology, node/value registration, and use-def consistency.
- `ir.Node` and `ir.Value` are not frozen dataclasses.
- `ir.TensorType` remains an immutable value object.
- Public convenience accessors such as `value.producer`, `value.users`, `node.inputs`, and `node.outputs`
	remain part of the developer-facing API, but their consistency is maintained by `ir.Graph` rather than by
	circular constructor patterns.

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

## Ownership and Mutability

ProtoFX needs IR-level normalization and graph transforms before emission. That makes full-object immutability
the wrong default for graph structure.

The accepted ownership model is:

- `ir.Graph` owns node membership, value membership, topological order, and structural consistency.
- `ir.Node` and `ir.Value` are mutable IR entities whose updates happen through graph-aware APIs.
- `ir.TensorType` is an immutable value object attached to `Value` instances.
- `Value.producer` and `Value.users` are **read-only properties**. The underlying data (`_producer` and
  `_users`) is private and may only be written by `ir.Graph` methods. This enforces use-def consistency
  at the API level rather than relying on caller discipline.

This split is intentional.

- Graph structure is expected to change during import normalization and later analysis or rewrite passes.
- Tensor metadata behaves like plain value data and is safer to replace than to mutate in place.
- Producer/user relationships are structural invariants that must stay consistent with `Node.inputs` and
  `Node.outputs`. Making them read-only on `Value` prevents accidental desynchronization.

ProtoFX therefore does **not** use frozen dataclasses for `Node` and `Value` as an architectural constraint.
The previous frozen `Node.create()` factory pattern is superseded by graph-managed construction.

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

> **Implementation**: `ir.TensorType` (`src/protofx/ir/tensor_type.py`) holds `dtype: DType | None` and
> `shape: Shape`. `Shape` (`src/protofx/ir/shape.py`) is `tuple[Dim, ...] | None`, where
> `Dim` (`src/protofx/ir/dim.py`) is `int | str | None` — covering concrete, symbolic, and unknown dimensions
> respectively. `DType` (`src/protofx/ir/dtype.py`) is a backend-neutral enum whose integer values mirror
> `onnx.TensorProto.DataType`.

### 7. Tensor Metadata Lives on Values

`TensorType` is attached to `Value`, not to `Node`.

This keeps type information aligned with actual data flow and avoids duplicating metadata across producers and
consumers.

> **Implementation**: `ir.TensorType` is a frozen dataclass. Conversion between ONNX / PyTorch dtypes and
> `ir.DType` is handled by `utils.dtype.onnx_dtype_to_ir()` and `utils.dtype.ir_dtype_to_torch()` —
> keeping the IR layer free of `onnx` and `torch` dependencies.

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

### 14. Graph Owns Structural Consistency

Producer links, user links, node ordering, and graph membership are owned by `ir.Graph`.

`Node` and `Value` may expose convenience accessors for these relationships, but `Graph` is the component that
must create, update, and validate them.

This rule exists to avoid distributing structural invariants across several mutable objects.

### 15. Public API Convenience Does Not Imply Distributed Ownership

ProtoFX keeps direct-feeling developer APIs where they help readability.

For example, code may still read:

- `value.producer`
- `value.users`
- `node.inputs`
- `node.outputs`

However, those conveniences do not make `Value` or `Node` the owners of graph consistency. Ownership remains in
`Graph`, which is the only place allowed to perform structural mutations without breaking invariants.

---

## Implementation Notes

This section records design decisions made during implementation of the IR types.

### DType Enum

`ir.DType` is a Python `enum.Enum` whose integer values are identical to `onnx.TensorProto.DataType`.
This alignment allows the importer to convert with `DType(elem_type)` without a lookup table, while the IR
itself has zero dependency on the `onnx` package.

Types not yet supported by the IR (e.g. `INT4`, `UINT4`, `FLOAT4E2M1`) map to `None` at import time and can
be added to the enum as PyTorch gains native support for them.

### Dim and Shape

- `Dim = int | str | None` — concrete, symbolic, or unknown dimension.
- `Shape = tuple[Dim, ...] | None` — `None` means entirely unknown rank and shape.

Symbolic dimensions are represented as plain `str` values (e.g. `"batch"`, `"seq_len"`).  This is sufficient
for Milestone 1; compatibility with `torch.export` `SymInt` may require a richer representation later.

### Dtype Mapping Utilities

Two functions in `utils.dtype` bridge the IR boundary:

| Function | Direction | Dependency |
|----------|-----------|------------|
| `onnx_dtype_to_ir(elem_type)` | ONNX → IR | `onnx` (importer side) |
| `ir_dtype_to_torch(dtype)` | IR → PyTorch | `torch` (emitter side, lazy import) |

`torch` is imported lazily inside `ir_dtype_to_torch` to keep import time fast for modules that only need
the IR layer.

### Value and ValueKind

`ir.Value` (`src/protofx/ir/value.py`) is a mutable IR entity representing one data-flow object in a graph.

The expected field shape is:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | — | Stable internal identifier, assigned externally by the graph owner |
| `kind` | `ValueKind` | — | Origin classification (see below) |
| `tensor_type` | `TensorType` | — | Tensor metadata (dtype + shape) |
| `name` | `str \| None` | `None` | Original ONNX name preserved as source metadata |
| `producer` | `Node \| None` | `None` | Producing node, managed by Graph |
| `users` | `list[tuple[Node, int]]` | `[]` | Consumer (node, slot) pairs, managed by Graph |

`ir.ValueKind` is an `enum.Enum` with `auto()` values:

- `GRAPH_INPUT` — a runtime graph input
- `NODE_OUTPUT` — an output produced by an IR node
- `SENTINEL` — a placeholder for an omitted optional ONNX input
- `CONSTANT` — a constant produced by a `Constant` op during import
- `INITIALIZER` — a graph-level initializer (pretrained weight, etc.)

**Mutability**: `Value` is intentionally not frozen. Import normalization and later graph transforms need to
update producer links, user links, names, and tensor metadata without replacing the entire object identity.

`tensor_type` updates should still prefer replacement of the `TensorType` instance rather than in-place
mutation of tensor metadata internals.

**Identity**: `id` uniqueness is not enforced by `Value` itself. The graph owner (`ir.Graph`) is
responsible for ensuring all `Value` ids are unique within a graph.

**Kind comparison**: callers compare kinds directly (`value.kind == ValueKind.SENTINEL`) rather than using
helper properties. This keeps the `Value` API surface minimal and explicit.

**Ownership**: `Value` does not own consistency of its `producer` or `users` relationships. Those are managed by
`Graph` mutation APIs.

### Node and AttributeValue

`ir.Node` (`src/protofx/ir/node.py`) is a mutable IR entity representing one normalized operation.

The expected field shape is:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | — | Stable internal identifier, assigned externally by the graph owner |
| `op_type` | `str` | — | ONNX operator type (e.g. `"Relu"`, `"Conv"`) |
| `inputs` | `list[Value]` | — | Ordered input `Value` references preserving ONNX positional order |
| `outputs` | `list[Value]` | `[]` | Ordered output `Value` references, one per operator output |
| `domain` | `str` | `""` | ONNX operator domain (empty string = default domain) |
| `opset_version` | `int \| None` | `None` | ONNX opset version for this node |
| `attributes` | `dict[str, AttributeValue]` | `{}` | Normalized Python-native attributes |
| `name` | `str \| None` | `None` | Original ONNX node name preserved as source metadata |

**AttributeValue type alias**: `int | float | str | bytes | list[int] | list[float] | list[str] | list[bytes]`.
All ONNX attributes are normalized into these Python-native forms during import. The emitter must not depend on
raw `onnx.AttributeProto` structures. Tensor-typed attributes (e.g. constant `value` on `Constant` nodes) are
not yet included and will be added when the importer is developed.

**Construction model**: `Node` is not responsible for atomically resolving `Node ↔ Value` circular references.
That responsibility belongs to `Graph`, which creates nodes, registers outputs, updates producer links, and
maintains use-def consistency as one graph-level operation.

This replaces the earlier frozen `Node.create()` factory pattern. Graph-aware construction is preferred because
ProtoFX expects normalization and transform passes to edit graph structure after import.

**Identity**: like `Value`, `id` uniqueness is not enforced by `Node` itself. The graph owner (`ir.Graph`)
is responsible for ensuring all `Node` ids are unique within a graph.

Normalization may unify how their values are accessed, but the graph boundary contract must preserve which
values are runtime inputs and which are statically bound constants.

### Graph

`ir.Graph` (`src/protofx/ir/graph.py`) is the structural owner of all `Node` and `Value` instances.

**Construction**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str \| None` | `None` | Optional graph name for diagnostics |
| `parent` | `Graph \| None` | `None` | Reserved for future subgraph support |

**State**:

| Field | Type | Description |
|-------|------|-------------|
| `inputs` | `list[Value]` | Ordered graph input values |
| `outputs` | `list[Value]` | Ordered graph output values |
| `nodes` | `list[Node]` | Nodes in insertion order |

Internal registries (`_values`, `_nodes`) and auto-ID counters (`_next_value_id`, `_next_node_id`) are private.

**Construction APIs**:

- `add_input(tensor_type, *, name=None)` — create a `GRAPH_INPUT` value, register it, and append to `inputs`.
- `make_node(op_type, inputs, output_types, *, domain="", opset_version=None, attributes=None, name=None)` —
  create a `Node` with `NODE_OUTPUT` values, wire producer and user links, and register everything.

**Mutation APIs** (all mutations go through `Graph` to maintain use-def consistency):

- `set_node_inputs(node, new_inputs)` — atomically rewire: (1) remove old users, (2) replace `node.inputs`,
  (3) add new users.
- `set_value_type(value, tensor_type)` — update tensor metadata on a value.
- `set_graph_outputs(outputs)` — replace the graph output list.
- `remove_node(node)` — fast-fail with `ValueError` if any output value has consumers; otherwise clean up
  users, unregister values, and remove from node list.

**Analysis APIs**:

- `topological_sort()` — return nodes in topological order using Kahn's algorithm. Raises `ValueError` on cycle.
- `validate()` — check all IR invariants: producer back-references, input registration, use-def consistency,
  and acyclicity.

**ID generation**: Graph auto-generates IDs: `v0, v1, ...` for values, `n0, n1, ...` for nodes.

### 16. Control-Flow Readiness Without Early Over-Commitment

Milestone 1 invariants should preserve future extensibility for control flow, but they do not yet define a full
region or subgraph ownership model.

ProtoFX should avoid baking in assumptions that would prevent `If`, `Loop`, or `Scan`, while also avoiding a
premature full control-flow abstraction before those operators are designed in detail.

### 17. Strict Validation Policy

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

### 18. Invariants Before Field-Level API

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
- graph-aware mutation APIs for construction, normalization, and refactoring

The graph must be simple to traverse without access to ONNX proto internals.

`Graph` is the sole owner of structural consistency. It is responsible for operations such as:

- creating nodes and output values together
- replacing an input or all uses of a value
- inserting, moving, and erasing nodes
- updating producer and user links
- validating topological order after mutation

### Value

`ir.Value` is the central data-flow object.

It should contain:

- a stable identifier or name
- a graph-backed reference to its producer node, if any
- a graph-backed list or view of user nodes
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
- ordered input `Value` references exposed through graph-managed APIs
- ordered output `Value` references exposed through graph-managed APIs
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
3. **`Graph` ownership model and mutation API**
4. **Refactor `Value` around graph-managed relationships**
5. **Refactor `Node` around graph-managed relationships**
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

### 3. `Graph` ownership model and mutation API

`Graph` must be designed before the final `Node` and `Value` refactor because it owns structural consistency.

This step should define:

- node and value registration rules
- graph input and output storage
- topological ordering guarantees
- producer and user bookkeeping
- mutation primitives such as node insertion, removal, and use replacement

Without this step, `Node` and `Value` drift toward self-managed relationships and circular construction.

### 4. Refactor `Value` around graph-managed relationships

Once `Graph` ownership is defined, `Value` should be reshaped into a lightweight mutable entity.

This refactor should:

- remove frozen-dataclass assumptions
- keep `id`, `kind`, `tensor_type`, and `name` as direct fields
- expose `producer` and `users` as graph-consistent accessors or graph-managed fields
- avoid copy-on-write update patterns such as `dataclasses.replace()` for structural edits

### 5. Refactor `Node` around graph-managed relationships

After `Value` is aligned to graph ownership, `Node` should be refactored to remove constructor-time circularity.

This refactor should:

- remove the frozen `Node.create()` factory pattern
- keep `op_type`, `domain`, `opset_version`, `attributes`, and `name` as direct node metadata
- expose ordered inputs and outputs without making `Node` the owner of use-def consistency
- make node creation a graph-level operation rather than a dataclass trick

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

## Refactoring Plan from the Current Implementation

ProtoFX already has `TensorType`, `Value`, and `Node` skeletons in place. The current implementation uses frozen
dataclasses for `Value` and `Node`, and a `Node.create()` factory to work around constructor-time circular
references. That design is a reasonable bootstrap, but it is not the accepted long-term architecture.

The refactor plan is:

1. Introduce `ir.Graph` as the first-class owner of nodes, values, and topological order.
2. Move node creation from `Node.create()` into graph-managed construction APIs.
3. Remove frozen semantics from `Value` and `Node`.
4. Replace copy-on-write mutation patterns with graph-aware updates.
5. Preserve convenient read APIs on `Node` and `Value` so downstream importer, validation, and emitter code stays
	readable.
6. Add validation that checks graph membership, producer/user consistency, and topological order after mutation.
7. Update tests to assert graph-level invariants rather than frozen-instance behavior.

### Immediate Code Implications

The current codebase should expect the following concrete changes:

- tests that assert `FrozenInstanceError`-style behavior for `Node` and `Value` will be removed or rewritten
- tests around `Node.create()` will move to `Graph` construction tests
- `Value.producer` and future `Value.users` behavior will be validated through graph-managed edits
- importer code should target `Graph.add_node(...)`-style APIs instead of building partially initialized objects

### Deferred Work

The following work is intentionally deferred until the graph-owned model is in place:

- generic pass-manager infrastructure
- advanced rewrite libraries
- full control-flow region ownership
- symbolic shape constraint solving beyond `TensorType`

## Summary

ProtoFX IR is not an optional abstraction layer added for theoretical purity.

It is the normalization boundary that turns ONNX serialization-oriented structures into a simple internal graph
model suitable for validation and `torch.fx` emission. For ProtoFX, that boundary remains intentionally thin,
but it is now explicitly graph-owned, mutable where graph structure requires it, and designed for IR-level
normalization and transformation.
