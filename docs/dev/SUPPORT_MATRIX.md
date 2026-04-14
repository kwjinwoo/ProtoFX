# Support Matrix

This document provides an at-a-glance summary of ProtoFX's current validated model coverage and downstream-task
coverage for repository readers.

The architectural decision behind this visibility surface is recorded in
`docs/adr/0006-coverage-visibility-support-matrix.md`.
Detailed validation boundaries remain in `docs/dev/MODEL_VALIDATION.md` and
`docs/dev/DOWNSTREAM_VALIDATION.md`.
Milestone planning remains in `docs/ROADMAP.md`.

## Purpose

Repository readers often need a fast answer to two questions:

1. Which reference models are currently covered?
2. Which downstream PyTorch tasks are validated for those exact models?

This document answers those questions without replacing the authoritative validation suites.

## Source Of Truth

- `tests/models/manifests/` declares the current reference-model set.
- `tests/models/` owns ONNX Runtime parity claims for those reference models.
- `tests/downstream/` owns downstream PyTorch-tooling compatibility claims.
- `docs/dev/OPSET_COMPATIBILITY.md` remains the op-level matrix. This document focuses on model and downstream-task
  coverage.

If this document disagrees with those sources, the suites and manifests win.

## Status Vocabulary

| Status | Meaning |
|--------|---------|
| `Validated` | The exact model-task pair has authoritative pytest coverage today. |
| `Synthetic only` | The task is validated on representative emitted graphs, but not yet on that exact reference model. |
| `Not yet model-validated` | No authoritative test exists for that exact model-task pair. Do not infer support. |
| `Planned` | The roadmap names this area as future work, but it is not current validated coverage. |

## Family Rollup

| Family Group | Current Validated Models | Current Model-Level Downstream Validation | Roadmap Expansion |
|--------------|--------------------------|-------------------------------------------|-------------------|
| Smoke / baseline | SqueezeNet (`squeezenet1_0`) | `torch.compile`, PT2E quantization | Retain as the fastest end-to-end gate |
| Vision | ResNet18, ViT-B/16 | `torch.compile` on ResNet18 | ResNet50, EfficientNet, MobileNet, ... |
| NLP | BERT | `torch.compile` on BERT | GPT-2, RoBERTa, DistilBERT, ... |
| Multi-modal | None today | None today | CLIP, ... |

## Current Model-By-Task Matrix

| Model | Family Group | ONNX Runtime Parity | `torch.compile` | PT2E Quantization | `torch.export` | Custom FX Pass |
|-------|--------------|---------------------|-----------------|-------------------|----------------|----------------|
| SqueezeNet (`squeezenet1_0`) | Smoke / baseline | `Validated` | `Validated` | `Validated` | `Synthetic only` | `Synthetic only` |
| ResNet18 | Vision | `Validated` | `Validated` | `Synthetic only` | `Synthetic only` | `Synthetic only` |
| ViT-B/16 | Vision | `Validated` | `Synthetic only` | `Synthetic only` | `Synthetic only` | `Synthetic only` |
| BERT | NLP | `Validated` | `Validated` | `Synthetic only` | `Synthetic only` | `Synthetic only` |

## Interpretation Rules

- `Validated` means the current repository contains authoritative coverage for that exact pair. It does not imply
  exhaustive validation across every platform, backend, or exporter configuration.
- For downstream tasks, supported-environment guarantees remain exactly those defined in
  `docs/dev/DOWNSTREAM_VALIDATION.md`.
- `Synthetic only` is intentionally weaker than model-level validation. It means the task is exercised on
  representative emitted graphs, not on the exact named reference model in this matrix.
- Planned models do not appear in the matrix until manifests and authoritative tests exist.
