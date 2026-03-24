# IR Pipeline Contracts

This document records the boundaries around the ProtoFX IR.

## Pipeline Boundary

The intended conversion pipeline is:

```text
onnx.ModelProto
  -> importer
  -> ir.Graph
  -> validation / analysis passes
  -> emitter
  -> torch.fx.GraphModule
```

## Importer Contract

The importer is responsible for ONNX-aware parsing and normalization.

It must:

- parse ONNX protobuf structures
- resolve opset and domain differences
- normalize attributes into Python-native values
- normalize constants, initializers, and omitted optional inputs into IR forms
- preserve source provenance needed for diagnostics
- produce graph-valid IR or fail early

The importer must not leak raw ONNX protobuf handling into the emitter.

## Validation and Analysis Contract

Validation targets normalized IR, not raw ONNX inputs.

Validation is responsible for:

- graph well-formedness
- producer and user consistency
- ordered interface consistency
- required attribute presence and normalized form
- shape and dtype constraints when enough metadata is available

Unknown metadata remains explicit and valid when the source model does not provide enough information.

## Emitter Contract

The emitter is responsible for FX-aware lowering from normalized IR.

It must:

- consume normalized `ir.Graph` structures rather than raw ONNX nodes
- build `torch.fx.Graph` and `torch.fx.GraphModule`
- delegate operator-specific lowering through the op handler registry
- keep `torch` imports lazy where practical

The emitter must not reinterpret raw ONNX protobuf details.

## Non-Goals

The IR boundary does not imply:

- a full compiler optimization framework
- backend-specific lowering policy embedded in the IR core
- aggressive graph rewriting as a Milestone 1 requirement
