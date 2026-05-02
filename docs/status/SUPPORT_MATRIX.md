---
schema_version: 1
doc_kind: status
title: Support matrix
summary: Representative snapshot of current validation anchors across model families and downstream tooling.
authority: derived
keywords: [status, support, coverage, validation]
source_of_truth:
  - tests/models/manifests/
  - tests/models/
  - tests/downstream/
  - docs/status/OPSET_COMPATIBILITY.md
related_docs:
  - docs/dev/MODEL_VALIDATION.md
  - docs/dev/DOWNSTREAM_VALIDATION.md
  - docs/adr/0007-agent-facing-documentation-entrypoint-and-summary.md
---

# Support Matrix

<!-- section:scope -->
## Scope

This document provides a representative, agent-oriented snapshot of ProtoFX validation coverage across model
families and downstream-tooling gates.

<!-- section:snapshot-semantics -->
## Snapshot Semantics

- This page is intentionally representative rather than exhaustive.
- Named examples are coverage anchors, not the full supported set.
- Exact current coverage belongs to manifests and pytest suites, not to this summary page.
- Planned work belongs in `docs/ROADMAP.md`, not in this snapshot.

<!-- section:current-state -->
## Current State

| Area | Representative exact coverage today | Representative synthetic-only coverage | Check here for exact current status |
|------|-------------------------------------|----------------------------------------|-------------------------------------|
| Smoke / baseline | SqueezeNet model parity, `torch.compile`, and PT2E quantization | `torch.export` and custom FX-pass smoke coverage | `tests/models/manifests/smoke/`, `tests/models/`, `tests/downstream/` |
| Vision | ResNet18 model parity and `torch.compile`; additional vision parity coverage exists in the current manifests and suites | `torch.export`, PT2E quantization, and custom FX-pass coverage remain representative gates here | `tests/models/manifests/vision/`, `tests/models/`, `tests/downstream/` |
| NLP | BERT and GPT2Model model parity and `torch.compile`; additional NLP parity coverage exists in the current manifests and suites | `torch.export`, PT2E quantization, and custom FX-pass coverage remain representative gates here | `tests/models/manifests/nlp/`, `tests/models/`, `tests/downstream/` |
| Multi-modal | CLIP model parity | No model-level downstream validation is summarized here | `tests/models/manifests/multimodal/`, `tests/models/`, `tests/downstream/` |

<!-- section:source-of-truth -->
## Source of Truth

- `tests/models/manifests/` declares the exact current reference-model set.
- `tests/models/` owns ONNX Runtime parity claims for those reference models.
- `tests/downstream/` owns downstream PyTorch-tooling compatibility claims.
- `docs/status/OPSET_COMPATIBILITY.md` summarizes op-level registry coverage.

If this document disagrees with those sources, the suites, manifests, and generated matrix win.

<!-- section:limitations -->
## Limitations

- Omitted models or tasks are not negative claims.
- This page does not prove exhaustive support coverage.
- This page does not widen any contract defined in `docs/dev/`.

<!-- section:references -->
## References

- `docs/dev/MODEL_VALIDATION.md`
- `docs/dev/DOWNSTREAM_VALIDATION.md`
- `docs/ROADMAP.md`
