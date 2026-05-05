---
schema_version: 1
doc_kind: dev
title: IR invariants
summary: Normative structural and semantic invariants for ProtoFX IR, including recursive control-flow child-graph validation rules.
authority: authoritative
keywords: [ir, invariants, validation, control-flow]
source_of_truth:
  - src/protofx/ir/
  - src/protofx/importers/
  - tests/ir/
  - tests/importer/
related_docs:
  - docs/adr/0001-thin-graph-owned-ir.md
  - docs/adr/0008-control-flow-subgraphs-and-if-mvp.md
  - docs/dev/ir/control-flow.md
---

# IR Invariants

<!-- section:purpose -->
## Purpose

This document defines the accepted structural and semantic invariants for ProtoFX IR.

<!-- section:scope -->
## Scope

These invariants apply to root graphs and child graphs produced through the control-flow extension.

<!-- section:contract -->
## Contract

These invariants are implementation-facing contracts derived from ADR-0001 and ADR-0008.

<!-- section:invariants -->
## Invariants

### Structural Invariants

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

Original ONNX names should be preserved as source metadata when available, but internal graph correctness must not
depend on them.

### 4. Single Producer Invariant

Every `Value` whose kind is `NODE_OUTPUT` must have exactly one producer.

Values whose kind is `GRAPH_INPUT`, `INITIALIZER`, `CONSTANT`, or `SENTINEL` must not have a producer
(`producer=None`). The importer inlines ONNX `Constant` ops into producerless `CONSTANT` values; constants that
remain as nodes produce `NODE_OUTPUT` values instead.

### 5. Ordered Inputs and Outputs

Node inputs and outputs preserve ONNX positional order.

### 6. Acyclic Graph with Explicit Topological View

Each graph boundary must remain acyclic, but `ir.Graph.nodes` is not required to stay physically topologically
ordered after every mutation.

When consumers require dependency-safe ordering, they must use `Graph.topological_sort()`.

### 7. Graph Inputs and Initializers Remain Distinct

Graph inputs and initializers are semantically distinct and must remain distinct in the IR.

### 8. Graph Owns Structural Consistency

Producer links, user links, node ordering, and graph membership are owned by `ir.Graph`.

### 9. Public API Convenience Does Not Imply Distributed Ownership

Direct-feeling accessors such as `value.producer`, `value.users`, `node.inputs`, and `node.outputs` do not make
those objects the owners of graph consistency.

### 10. Child Subgraphs Are Structural Node Data

Control-flow child graphs are structural node-owned data and do not belong in the scalar `AttributeValue` space.

### 11. Child-Graph Ownership Is Graph-Local

Each child graph owns its own inputs, outputs, initializers, node list, value registry, and topological traversal.

The `parent` link records enclosure; it does not merge ownership domains.

### 12. Outer-Scope Capture Is Explicit

After import normalization, a child graph must not depend on hidden outer-scope values.

Any captured value required by a child graph is represented as an explicit child-graph input.

### Semantic Invariants

### 13. Constant Interface with Source Preservation

Initializers and `Constant` op outputs share a common constant interface inside the IR, while preserving their source
kind as metadata.

### 14. Explicit Omitted Optional Inputs

An omitted optional ONNX input is represented by a dedicated sentinel `Value`.

The sentinel must be distinguishable from ordinary runtime values.

### 15. Attribute Normalization Happens in the Importer

All scalar and list-like node attributes are normalized into Python-native forms during import.

The emitter must not depend on raw ONNX `AttributeProto` structures.

### 16. Source Provenance Is Preserved

The IR should retain enough source metadata to produce useful diagnostics and debugging output.

At minimum, preserve when available:

- graph name
- original node names
- output names
- operator type and domain
- opset context

The control-flow extension does not require separate provenance objects beyond these graph and node fields.

### 17. Tensor Metadata Lives on Values

`TensorType` is attached to `Value`, not to `Node`.

### 18. Rich Unknown Representation for Tensor Metadata

`TensorType` must distinguish at least:

- unknown shape
- scalar shape
- empty dimensions
- partially known symbolic dimensions
- fully known shapes

### 19. Branch Arity Is Strict Where the Operator Contract Requires It

For `If`, then and else branch output arity must match exactly.

### 20. Metadata Validation Is Conditional, Not Speculative

When enough metadata is available, validation should enforce compatible dtype and shape expectations.

When the source model does not provide enough information, the IR must preserve that uncertainty instead of
guessing.

### 21. Control-Flow Extensibility Avoids Early Region-IR Commitment

The accepted control-flow extension must preserve nested-graph support without forcing ProtoFX into a full region
model.

### 22. Strict Validation Policy

When the importer produces invalid IR, ProtoFX should fail early rather than defer structural or semantic problems
to the emitter.

The minimum enforcement point is that the importer returns only graphs that pass `graph.validate()`, including any
recursively imported child graphs.

<!-- section:failure-semantics -->
## Failure Semantics

- Violated invariants must fail during validation rather than being silently repaired.
- Unsupported control-flow forms remain explicit failures rather than partial success with dropped structure.

<!-- section:non-goals -->
## Non-Goals

- Defining `Loop` and `Scan` semantics in this invariant set
- Requiring global topological order across parent and child graph boundaries
- Guessing missing tensor metadata

<!-- section:references -->
## References

- Related code: `src/protofx/ir/graph.py`, `src/protofx/importers/_onnx.py`
- Related tests: `tests/ir/`, `tests/importer/`
- Related ADRs: `docs/adr/0001-thin-graph-owned-ir.md`, `docs/adr/0008-control-flow-subgraphs-and-if-mvp.md`
