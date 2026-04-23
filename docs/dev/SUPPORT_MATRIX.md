# Support Matrix

This document provides a representative, agent-oriented snapshot of ProtoFX validation coverage.

The architectural decision behind this visibility model is recorded in
`docs/adr/0007-agent-facing-documentation-entrypoint-and-summary.md`.
Detailed validation boundaries remain in `docs/dev/MODEL_VALIDATION.md` and
`docs/dev/DOWNSTREAM_VALIDATION.md`.
Milestone planning remains in `docs/ROADMAP.md`.

## Purpose

Repository readers often need a fast answer to two narrower questions:

1. Which coverage anchors exist today?
2. Where should I look for the exact current status?

This document answers those questions without pretending to be an exhaustive live coverage database.

## How To Read This Document

- This page is intentionally representative rather than exhaustive.
- Named examples are coverage anchors, not the full supported set.
- Exact current coverage belongs to manifests and pytest suites, not to this summary page.

## Authoritative Sources

- `tests/models/manifests/` declares the exact current reference-model set.
- `tests/models/` owns ONNX Runtime parity claims for those reference models.
- `tests/downstream/` owns downstream PyTorch-tooling compatibility claims.
- `docs/dev/OPSET_COMPATIBILITY.md` remains the op-level compatibility matrix.

If this document disagrees with those sources, the suites and manifests win.

## Representative Snapshot

| Area | Representative exact coverage today | Representative synthetic-only coverage | Check here for exact current status |
|------|-------------------------------------|----------------------------------------|-------------------------------------|
| Smoke / baseline | SqueezeNet model parity, `torch.compile`, and PT2E quantization | `torch.export` and custom FX-pass smoke coverage | `tests/models/manifests/smoke/`, `tests/models/`, `tests/downstream/` |
| Vision | ResNet18 model parity and `torch.compile`; additional vision parity coverage exists in the current manifests and suites | `torch.export`, PT2E quantization, and custom FX-pass coverage remain representative gates here | `tests/models/manifests/vision/`, `tests/models/`, `tests/downstream/` |
| NLP | BERT model parity and `torch.compile`; additional NLP parity coverage exists in the current manifests and suites | `torch.export`, PT2E quantization, and custom FX-pass coverage remain representative gates here | `tests/models/manifests/nlp/`, `tests/models/`, `tests/downstream/` |
| Multi-modal | CLIP model parity | No model-level downstream validation is summarized here | `tests/models/manifests/multimodal/`, `tests/models/`, `tests/downstream/` |

## Interpretation Rules

- A named exact model in this document means the repository contains authoritative pytest coverage for that
  representative case today.
- `Synthetic only` means the task is exercised on representative emitted graphs, not on the exact named reference
  model in this summary.
- Omitted models or tasks are not negative claims. Check the manifests, suites, and roadmap directly.
- Planned work belongs in `docs/ROADMAP.md`, not in this summary page.
