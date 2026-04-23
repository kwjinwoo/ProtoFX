# Architecture

## Overview

ProtoFX is a three-stage pipeline that converts ONNX models into PyTorch `torch.fx.GraphModule` objects.

```
ONNX ModelProto ──▶ Importer ──▶ Thin Normalized IR ──▶ Validation / Analysis ──▶ Emitter ──▶ torch.fx.GraphModule
```

Accepted architecture decisions are recorded in `docs/adr/`.
Detailed subsystem specifications live under `docs/dev/`.
Public API reference lives under `docs/src/`.
For agent-oriented navigation across the documentation tree, start at `docs/README.md`.
Milestone 1 IR contract reconciliation is recorded in `docs/adr/0003-milestone-1-ir-contract-reconciliation.md`.

## Documentation System

ProtoFX uses a layered documentation model so architecture decisions, specifications, and planning state do
not collapse into a single document.

| Document area | Purpose | Authority |
|---------------|---------|-----------|
| `docs/README.md` | Agent-facing navigation and document selection | Navigation only; not an authority surface |
| `docs/adr/` | Records accepted architecture decisions and their rationale | Source of truth for structural decisions |
| `docs/dev/` | Records technical specifications derived from accepted decisions | Source of truth for implementation-facing contracts |
| `docs/src/` | Records public API reference and module-level usage notes | Reference surface for exported APIs |
| `docs/ROADMAP.md` | Records milestones, priorities, and project-level sequencing | Source of truth for planned project scope |
| Temporary workboard (when present) | User-maintained execution checklist for directing agents | Convenience only; not authoritative for architecture |

When a document category disagrees with another, the precedence is:

1. ADRs for architectural decisions.
2. Development specifications for detailed contracts.
3. Roadmap for milestone priority.
4. Temporary workboard guidance, if a workboard exists.

## Directory Structure

```
src/protofx/
├── importers/       # ONNX graph → IR (intermediate representation)
├── ir/              # Internal graph IR nodes and types
├── emitters/        # IR → torch.fx Graph construction
├── ops/             # Per-ONNX-op conversion handlers (one file per domain)
└── utils/           # Shared helpers (shape inference, type mapping)
```

## Components

### Importers

Reads `onnx.ModelProto` and converts it into the internal IR.

- Parses nodes, tensors, and initializer values from the ONNX graph
- Handles branching based on ONNX opset version
- Recursively imports subgraphs (control flow)

### IR (Intermediate Representation)

A thin normalized graph representation independent of both ONNX and `torch.fx`.

- Holds normalized node, value, constant, and tensor type information
- Uses `ir.Graph` as the owner of topology and use-def relationships
- Keeps `ir.Node` and `ir.Value` mutable enough for normalization and graph transforms
- Preserves convenient relationship accessors (`value.producer`, `value.users`, `node.inputs`, `node.outputs`)
- Acts as the semantic boundary between Importer and Emitter
- Provides a stable target for validation and analysis before backend emission
- Exposes `Graph.topological_sort()` as the authoritative dependency-safe node ordering view
- See `IR.md` and the `ir/` specification documents for details

### Emitters

Traverses the IR and creates nodes in `torch.fx.Graph`.

- Constructs `torch.fx.Graph` and `torch.fx.GraphModule`
- Converts each IR node to an FX node via the op handler registry
- Consumes normalized IR data rather than raw ONNX protobuf structures
- Keeps `torch` imports lazy to optimize import speed

### Ops

Contains per-ONNX-op conversion handlers.

- Registered with the `@register_op("opname")` decorator
- Organized by domain (e.g., `nn.py`, `math.py`)
- Each handler returns `torch.fx.Node`(s)

### Utils

Shared helper modules.

- Shape inference utilities
- ONNX ↔ PyTorch type mapping
- Tensor conversion helpers

## Validation Structure

ProtoFX treats validation as part of the architecture, not as an ad hoc scripting concern.

| Suite | Role |
|-------|------|
| `tests/importer/` | Verifies ONNX parsing, normalization, and importer boundary behavior. |
| `tests/emitter/` | Verifies FX emission behavior, value-kind lowering, and end-to-end emission smoke coverage. |
| `tests/ir/` | Verifies the graph-owned IR model, mutation APIs, and invariants. |
| `tests/utils/` | Verifies shared boundary helpers such as dtype conversion utilities. |
| `tests/ops/` | Verifies handler behavior and emitted FX structure with small targeted fixtures. |
| `tests/parity/` | Verifies op-level ONNX Runtime parity using synthetic models built in code. |
| `tests/models/` | Verifies manifest-driven reference-model parity; exact current model coverage is declared by manifests and suite contents. |
| `tests/downstream/` | Verifies representative compatibility of emitted `GraphModule` objects with downstream PyTorch tooling such as `torch.compile`, `torch.export`, FX quantization, and custom FX passes. |

Reference-model validation follows a different asset boundary than small parity tests.

- Large ONNX artifacts are externalized and materialized into a cache outside the git worktree.
- The repository stores human-reviewable manifests and tolerances rather than large vendored model binaries.
- Helper tooling may later exist under `scripts/` or another utility location, but pytest suites and their
	manifests remain the authoritative validation surface.

Downstream PyTorch-tooling validation follows a separate suite boundary from both ORT parity and
reference-model validation.

- Claims about `torch.compile`, `torch.export`, FX quantization, or custom FX-pass compatibility belong
	under `tests/downstream/`.
- Helper tooling may exist under `scripts/` or another utility location, but pytest suites remain the
	authoritative validation surface there as well.

See `docs/dev/MODEL_VALIDATION.md` for the detailed reference-model validation structure and asset policy.
See `docs/dev/DOWNSTREAM_VALIDATION.md` for the detailed downstream-tooling validation contract.
`docs/dev/SUPPORT_MATRIX.md` provides a derived representative summary of current validation anchors.
That summary is subordinate to the authoritative pytest suites and manifests above.

## Architectural Boundary

ProtoFX intentionally separates three concerns:

1. **Importer**: ONNX-aware parsing and normalization.
2. **IR and validation**: internal graph structure, metadata, and structural checks.
3. **Emitter**: FX-aware lowering from normalized IR.

This boundary is important for two reasons:

- ONNX protobuf details, opset quirks, and attribute decoding should not leak into FX emission code.
- FX-specific lowering decisions should not distort the imported graph model.

Within Milestone 1, the importer is expected to return validated IR. Fail-fast behavior belongs at the importer
boundary, while deeper analysis can remain a separate concern after import.

The project does **not** treat IR as a full compiler framework. It is a minimal normalization layer chosen to
support downstream compatibility, testing, and future expansion without over-design.

Within that boundary, ProtoFX treats `ir.Graph` as the structural owner of nodes, values, and topological
ordering. This avoids embedding graph consistency rules inside individual dataclass constructors while still
allowing a convenient object-oriented API for graph consumers.

Milestone 1 does not require `graph.nodes` itself to remain physically topologically sorted after every graph
mutation. Consumers that require dependency-safe order must use `Graph.topological_sort()`.

## Documentation Entry Point

The documentation tree is intentionally split by authority and purpose.

- Start at `docs/README.md` to choose the right document for the current question.
- Use `docs/adr/` for accepted decisions and rationale.
- Use `docs/dev/` for implementation-facing contracts and boundaries.
- Use `docs/src/` for public API reference.
- Use `docs/ROADMAP.md` for planned scope rather than current guarantees.
