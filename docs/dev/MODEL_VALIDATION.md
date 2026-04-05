# Model Validation

This document defines the structure, asset boundary, and cache policy for ProtoFX reference-model validation.

The architectural decision behind this policy is recorded in
`docs/adr/0004-externalized-reference-model-validation-assets.md`.
Milestone scope is tracked in `docs/ROADMAP.md`.

## Purpose

ProtoFX needs model-level validation in addition to op-level parity.

Op-level parity answers whether isolated handlers match ONNX Runtime on small synthetic graphs.
Reference-model validation answers whether the importer, IR normalization, and FX emitter still behave
correctly on representative end-to-end architectures.

These two validation layers serve different purposes and should not be collapsed into the same asset model.

## Validation Layers

| Layer | Scope | Asset source | Expected scale |
|-------|-------|--------------|----------------|
| `tests/ops/` | Handler semantics and FX lowering behavior | Small hand-authored graphs and fixtures | Small |
| `tests/parity/` | Synthetic ONNX vs. ONNX Runtime numerical parity | Code-generated ONNX models | Small |
| `tests/models/` | Standard model-family validation | Manifest-declared, cache-backed reference assets | Large |

## Repository Boundary

- Large reference ONNX binaries are not the normal git-tracked artifact for model-family validation.
- The repository should keep only human-reviewable declarations such as manifests, tolerances, and family
  coverage metadata.
- Small vendored ONNX fixtures remain acceptable only as narrow exceptions for bug regressions when
  code-generated or reproducibly exported alternatives are impractical.

This keeps source control focused on reviewable intent instead of generated binary payloads.

## Recommended Suite Structure

The recommended layout for Milestone 3 model-family validation is:

```text
tests/
├── parity/                # Synthetic, code-generated ONNX parity tests
└── models/                # Reference-model family validation
    ├── conftest.py        # Shared materialization, cache, and comparison fixtures
    ├── manifests/         # Human-reviewable model declarations
    │   ├── vision/
    │   └── nlp/
    ├── test_vision.py     # ResNet, ViT, and related vision families
    └── test_nlp.py        # BERT and related language families
```

This separation is intentional.

- `tests/parity/` stays optimized for small code-generated models and fast operator-focused feedback.
- `tests/models/` carries heavier end-to-end validation concerns, optional dependencies, asset caching, and
  broader family-level coverage.
- Helper scripts may assist materialization, but they do not replace the suite boundary above.

## Manifest Contract

Each reference-model declaration should capture enough information to materialize and validate a model
deterministically.

At minimum, a manifest should declare:

- a stable model identifier and family classification
- a canonical source location or reproducible export recipe
- an immutable revision, version, or content digest
- the expected opset and any export-time knobs that affect graph shape
- input signature metadata and representative input-generation rules
- output selection rules and numerical tolerances
- any required optional dependencies for materialization or execution

The manifest format itself should stay text-based and reviewable.

## Materialization and Cache Policy

Reference-model assets should be materialized into a cache outside the git worktree.

- Cache entries should be keyed by the declared model identity plus any source digest, revision, opset, and
  export settings that change the produced ONNX graph.
- Cached artifacts should be treated as disposable build products, not repository content.
- Cache population may happen through download, export, or conversion helpers, but the manifest remains the
  source of truth.
- The cache root must be configurable so local development and CI can point at different storage locations.
- Digest or revision mismatches must invalidate stale cache entries rather than silently reusing them.

## Execution Policy

Reference-model validation should be treated as an opt-in or separately gated suite rather than folded into
the same execution profile as small fast tests.

- Local development workflows may skip model-family tests when optional dependencies or materialized assets are
  unavailable.
- A dedicated CI job should materialize the declared assets and fail if required artifacts cannot be obtained
  or validated.
- Blocking CI may use a smaller smoke subset of manifests, while scheduled or manually triggered jobs may run a
  broader family matrix.

This preserves fast feedback for routine development without weakening the long-running validation contract.

## Relationship to Scripts

Helper tooling may exist for tasks such as downloading assets, exporting reference models, or prewarming a
cache. That tooling is useful, but it is not the architectural center of model validation.

ProtoFX treats the manifest-driven pytest suite as authoritative.
Scripts are support machinery; they are not the source of truth for coverage, pass/fail behavior, or reviewable
validation intent.
