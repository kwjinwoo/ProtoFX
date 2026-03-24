# IR Invariants

This document defines the accepted invariants for the initial ProtoFX IR.

These invariants are implementation-facing contracts derived from ADR-0001.

## Structural Invariants

### 1. Value-Centric Data Flow

All data-flow units are represented as `Value` objects.

This includes:

- graph inputs
- node outputs
- constants
- initializers
- explicit placeholders for omitted optional inputs

### 2. Multi-Output Nodes Produce Independent Values

A node with multiple outputs must produce one distinct `Value` per output.

Those outputs remain ordered according to the ONNX operator contract.

### 3. Stable Internal Identity

`Value` identity is based on stable internal identifiers, not directly on ONNX names.

Original ONNX names should be preserved as source metadata when available, but internal graph correctness must
not depend on them.

### 4. Single Producer Invariant

Every `Value` whose kind is `NODE_OUTPUT` must have exactly one producer.

Values whose kind is `GRAPH_INPUT`, `INITIALIZER`, `CONSTANT`, or `SENTINEL` must not have a producer
(`producer=None`). The importer inlines ONNX `Constant` ops into producerless `CONSTANT` values;
constants that remain as nodes produce `NODE_OUTPUT` values instead.

### 5. Ordered Inputs and Outputs

Node inputs and outputs preserve ONNX positional order.

### 6. Acyclic Graph with Explicit Topological View

The graph must remain acyclic, but `ir.Graph.nodes` is not required to stay physically topologically ordered
after every mutation.

When consumers require dependency-safe ordering, they must use `Graph.topological_sort()`.

### 7. Graph Inputs and Initializers Remain Distinct

Graph inputs and initializers are semantically distinct and must remain distinct in the IR.

### 8. Graph Owns Structural Consistency

Producer links, user links, node ordering, and graph membership are owned by `ir.Graph`.

### 9. Public API Convenience Does Not Imply Distributed Ownership

Direct-feeling accessors such as `value.producer`, `value.users`, `node.inputs`, and `node.outputs` do not make
those objects the owners of graph consistency.

## Semantic Invariants

### 10. Constant Interface with Source Preservation

Initializers and `Constant` op outputs share a common constant interface inside the IR, while preserving their
source kind as metadata.

### 11. Explicit Omitted Optional Inputs

An omitted optional ONNX input is represented by a dedicated sentinel `Value`.

The sentinel must be distinguishable from ordinary runtime values.

### 12. Attribute Normalization Happens in the Importer

All node attributes are normalized into Python-native forms during import.

The emitter must not depend on raw ONNX `AttributeProto` structures.

### 13. Source Provenance Is Preserved

The IR should retain enough source metadata to produce useful diagnostics and debugging output.

At minimum, preserve when available:

- graph name
- original node names
- output names
- operator type and domain
- opset context

Milestone 1 does not require separate graph-boundary provenance objects beyond these fields.

### 14. Tensor Metadata Lives on Values

`TensorType` is attached to `Value`, not to `Node`.

### 15. Rich Unknown Representation for Tensor Metadata

`TensorType` must distinguish at least:

- unknown shape
- scalar shape
- empty dimensions
- partially known symbolic dimensions
- fully known shapes

### 16. Control-Flow Readiness Without Early Over-Commitment

Milestone 1 invariants should preserve future extensibility for control flow without defining a full region
model yet.

### 17. Strict Validation Policy

When the importer produces invalid IR, ProtoFX should fail early rather than defer structural or semantic
problems to the emitter.

For Milestone 1, the minimum enforcement point is that the importer returns only graphs that pass
`graph.validate()`.
