---
schema_version: 1
doc_kind: status
title: Symbolic shape propagation closure snapshot
summary: Derived Milestone 10 closure snapshot anchored to committed symbolic shape validation and propagation tests.
authority: derived
keywords: [status, ir, symbolic-shape, milestone-10]
source_of_truth:
  - tests/ir/test_shape.py
  - tests/ir/test_graph.py
  - tests/importer/test_onnx_importer.py
  - tests/ir/test_shape_propagation.py
related_docs:
  - docs/adr/0011-symbolic-shape-propagation-pipeline-and-validation-boundary.md
  - docs/dev/ir/contracts.md
  - docs/ROADMAP.md
---

# Symbolic Shape Propagation Closure Snapshot

<!-- section:scope -->
## Scope

This snapshot summarizes Milestone 10 closure evidence for symbolic shape propagation semantics based on committed
tests from Commits A and B.

<!-- section:snapshot-semantics -->
## Snapshot Semantics

- This page is derived evidence only and does not define new architecture or normative contracts.
- Authority remains ADR-0011 and `docs/dev/ir/contracts.md`.
- Milestone 10 closure state here is valid only to the extent directly anchored by committed tests.

<!-- section:current-state -->
## Current State

### Commit A evidence anchors

- Seed-only ONNX inference with tri-state compatibility semantics: covered.
  - `tests/ir/test_shape.py`
    - `TestCompareShapes::test_compare_both_unknown_shapes_is_unknown`
    - `TestCompareShapes::test_compare_unknown_and_known_shapes_is_unknown`
    - `TestCompareShapes::test_compare_symbolic_shapes_stays_unknown`
    - `TestCompareShapes::test_compare_symbolic_dim_with_known_mismatch_is_incompatible`
  - `tests/ir/test_graph.py`
    - `TestGraphControlFlowValidation::test_validate_skips_shape_mismatch_when_branch_shape_is_unknown`
    - `TestGraphLoopValidation::test_validate_skips_loop_carried_shape_mismatch_when_shape_unknown`
    - `TestGraphLoopValidation::test_validate_fails_for_loop_carried_shape_mismatch_when_provable`
    - `TestGraphScanValidation::test_validate_skips_scan_output_shape_mismatch_when_dim_unknown`
    - `TestGraphScanValidation::test_validate_fails_for_scan_output_shape_rank_mismatch_when_provable`
  - `tests/importer/test_onnx_importer.py`
    - `TestDeferredShapeBoundaries::test_import_model_rejects_loop_when_seed_shape_needs_propagation_fixup`
    - `TestDeferredShapeBoundaries::test_import_model_rejects_scan_when_seed_shape_needs_propagation_fixup`
    - `TestDeferredShapeBoundaries::test_shape_as_data_runtime_dynamic_lowering_is_explicitly_unimplemented`

### Commit B evidence anchors

- First-phase propagation coverage across target families: covered.
  - `tests/ir/test_shape_propagation.py`
    - `test_propagate_transpose_overrides_seed_metadata_with_symbolic_dim`
    - `test_propagate_unsqueeze_overrides_seed_metadata_with_partial_unknown_shape`
    - `test_propagate_concat_overrides_seed_metadata_with_symbolic_mismatch`
    - `test_propagate_conv_transpose_overrides_seed_metadata`
    - `test_propagate_maxpool_overrides_seed_metadata_with_symbolic_spatial_dim`
    - `test_propagate_average_pool_overrides_seed_metadata`
    - `test_propagate_global_average_pool_overrides_seed_metadata_with_partial_unknown`

### Existing supporting evidence (pre-Commit A/B; not a Commit A/B-added anchor)

- `tests/ops/test_control_flow_if.py::TestIfHandler::test_if_propagation_merges_branch_shape`

<!-- section:source-of-truth -->
## Source of Truth

- `tests/ir/test_shape.py`
- `tests/ir/test_graph.py`
- `tests/importer/test_onnx_importer.py`
- `tests/ir/test_shape_propagation.py`
- `docs/adr/0011-symbolic-shape-propagation-pipeline-and-validation-boundary.md`
- `docs/dev/ir/contracts.md`

If this snapshot conflicts with those files, tests and authoritative docs win.

<!-- section:limitations -->
## Limitations

- This snapshot is representative closure evidence, not exhaustive per-op proof.
- It does not claim Loop/Scan symbolic propagation extensions beyond current validation boundaries.
- It does not claim runtime dynamic lowering support for shape-as-data operators.
- It does not widen guarantees beyond ADR-0011 and `docs/dev/ir/contracts.md`.

<!-- section:references -->
## References

- `docs/adr/0011-symbolic-shape-propagation-pipeline-and-validation-boundary.md`
- `docs/dev/ir/contracts.md`
- `docs/ROADMAP.md`
- `tests/ir/test_shape.py`
- `tests/ir/test_graph.py`
- `tests/importer/test_onnx_importer.py`
- `tests/ir/test_shape_propagation.py`
