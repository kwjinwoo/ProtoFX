# Documentation Workboard

This workboard is a user-directed execution checklist for documentation gaps found in the current ProtoFX codebase.
It is not an architectural source of truth. Architectural authority remains with `docs/adr/`, `docs/dev/`, and
`docs/ROADMAP.md` as defined in `docs/dev/ARCHITECTURE.md`.

## Priority Table

| Priority | Document Layer | Gap | Why it matters | Primary references |
|----------|----------------|-----|----------------|--------------------|
| P0 | User doc | Root README documents a non-existent `protofx.to_fx()` entry point and an outdated `ctx`-style op-handler API. | New users will fail on first-run examples and extension guidance. | `README.md`, `docs/src/README.md`, `src/protofx/importers/__init__.py`, `src/protofx/emitters/__init__.py`, `src/protofx/ops/_registry.py` |
| P0 | User doc | Root README overstates downstream integration status and understates runtime requirements. | It advertises Milestone 4 capabilities as already available and mismatches package metadata. | `README.md`, `docs/ROADMAP.md`, `pyproject.toml` |
| P1 | API doc | Public API reference lacks dedicated pages for `protofx.ops` and `protofx.ir`. | Exported public symbols exist, but users have no authoritative reference for registry APIs or IR surface. | `docs/src/README.md`, `src/protofx/ops/__init__.py`, `src/protofx/ir/__init__.py` |
| P1 | Dev spec | Model-validation spec is conceptually correct but does not document the concrete manifest schema, optional extras, pytest marker usage, or cache behavior implemented in tests. | Contributors cannot reliably run or extend the reference-model suite from docs alone. | `docs/dev/MODEL_VALIDATION.md`, `tests/models/_manifest.py`, `tests/models/_cache.py`, `tests/models/conftest.py`, `pyproject.toml` |
| P1 | Dev spec | Validation/test architecture docs mention only `tests/ops/`, `tests/parity/`, and `tests/models/`, omitting importer, emitter, IR, and utility test suites. | Contributors lack guidance on where different categories of tests belong. | `docs/dev/ARCHITECTURE.md`, `tests/importer/`, `tests/emitter/`, `tests/ir/`, `tests/utils/` |
| P1 | Roadmap sync | Roadmap still marks the opset compatibility matrix as planned even though the generated document, script, and tests already exist. | Status drift weakens roadmap credibility and confuses milestone tracking. | `docs/ROADMAP.md`, `docs/dev/OPSET_COMPATIBILITY.md`, `scripts/gen_opset_matrix.py`, `tests/ops/test_compat_matrix.py` |
| P2 | ADR | No new architectural decision appears to be missing for the current documentation gaps. The main issue is synchronization of derived docs and user-facing docs. | This prevents unnecessary ADR churn while keeping attention on real documentation debt. | `docs/adr/README.md`, `docs/adr/0001-thin-graph-owned-ir.md`, `docs/adr/0002-documentation-system.md`, `docs/adr/0003-milestone-1-ir-contract-reconciliation.md`, `docs/adr/0004-externalized-reference-model-validation-assets.md` |

## Todo List

### P0 User Documentation

- [ ] Update the root README quick-start example to use the real public entry points (`import_model()` and `emit_graph()`) or explicitly document that no top-level convenience wrapper exists.
- [ ] Replace the outdated `ctx`-style op-handler example in the root README with the actual handler signature used by the registry.
- [ ] Rewrite README feature claims so downstream integration status matches the roadmap instead of implying Milestone 4 is already complete.
- [ ] Align README runtime requirements and installation guidance with `pyproject.toml`, especially the current PyTorch minimum version.

### P1 API Documentation

- [ ] Add a `docs/src/ops.md` reference page covering `register_op()`, `dispatch_op()`, `list_registry()`, handler signature, duplicate-registration behavior, and `opset_range` semantics.
- [ ] Add a `docs/src/ir.md` reference page covering the exported `protofx.ir` surface and linking each public type to the authoritative IR specification documents.
- [ ] Update `docs/src/README.md` so the API index links to the new `protofx.ops` and `protofx.ir` reference pages and keeps the top-level package status explicit.

### P1 Development Specifications

- [ ] Expand `docs/dev/MODEL_VALIDATION.md` with the concrete manifest schema currently enforced by `tests/models/_manifest.py`.
- [ ] Document model-validation execution prerequisites: optional dependency groups, the `model_validation` pytest marker, and how local runs differ from heavier CI coverage.
- [ ] Document cache behavior for reference-model materialization, including the default cache root, invalidation inputs, and the role of `export_kwargs` in cache identity.
- [ ] Update validation architecture docs so they describe the roles of `tests/importer/`, `tests/emitter/`, `tests/ir/`, and `tests/utils/` in addition to the larger validation layers.

### P1 Roadmap Synchronization

- [ ] Update `docs/ROADMAP.md` so the opset compatibility matrix status reflects the existing generated document, generation script, and test coverage.

### P2 ADR Review

- [ ] Confirm during documentation cleanup that no new ADR is required and that the work remains limited to derived-spec and user-document synchronization.
