# IR (Intermediate Representation)

The ProtoFX IR is a lightweight graph representation independent of both ONNX and `torch.fx`.

## Design Goals

- **Decoupling**: A neutral layer between Importer (ONNX-dependent) and Emitter (torch.fx-dependent)
- **Lightweight**: Holds only the minimal information required for conversion
- **Extensible**: Easy to add new metadata or analysis passes

## Core Components

| Component | Description |
|-----------|-------------|
| **Graph** | A DAG (Directed Acyclic Graph) composed of nodes and edges |
| **Node** | Represents a single operation, holding op type, inputs, outputs, and attributes |
| **Edge** | A connection representing data flow between nodes |
| **TensorType** | Tensor metadata such as shape and dtype |

## Conversion Flow

```
onnx.NodeProto  ──▶  ir.Node  ──▶  torch.fx.Node
onnx.GraphProto ──▶  ir.Graph ──▶  torch.fx.Graph
```
