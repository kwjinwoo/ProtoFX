# Model Validation

This document defines the structure, asset boundary, manifest schema, execution requirements, and cache
behavior for ProtoFX reference-model validation.

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
- The repository keeps only human-reviewable declarations such as manifests, tolerances, and family coverage
  metadata.
- Small vendored ONNX fixtures remain acceptable only as narrow exceptions for bug regressions when
  code-generated or reproducibly exported alternatives are impractical.

This keeps source control focused on reviewable intent instead of generated binary payloads.

## Current Suite Structure

The current `tests/models/` layout is:

```text
tests/models/
├── _cache.py
├── _manifest.py
├── conftest.py
├── manifests/
│   ├── nlp/
│   │   └── bert.yaml
│   ├── smoke/
│   │   └── smoke.yaml
│   └── vision/
│       ├── resnet18.yaml
│       └── vit_b_16.yaml
├── test_nlp.py
├── test_smoke.py
└── test_vision.py
```

- `_manifest.py` parses YAML into `ModelManifest` and enforces required fields and types.
- `_cache.py` materializes cache-backed ONNX exports from validated manifests.
- `conftest.py` provides the smoke fixtures and the shared `assert_model_parity()` helper.
- `test_smoke.py`, `test_vision.py`, and `test_nlp.py` are all gated with `@pytest.mark.model_validation`.

This separation is intentional.

- `tests/parity/` stays optimized for small code-generated models and fast operator-focused feedback.
- `tests/models/` carries heavier end-to-end validation concerns, optional dependencies, asset caching, and
  broader family-level coverage.
- Helper scripts may assist materialization, but they do not replace the suite boundary above.

## Manifest Contract

`load_manifest()` accepts only YAML mappings and validates the following schema.

| Field | Required | Type | Meaning |
|-------|----------|------|---------|
| `family` | Yes | `str` | Model family selector. The current builders support `torchvision` and `transformers`. |
| `model_name` | Yes | `str` | Model constructor or identifier within the selected family. |
| `opset` | Yes | `int` | ONNX opset passed to `torch.onnx.export()`. |
| `pretrained` | Yes | `bool` | Whether to load pretrained weights instead of seeded random initialization. |
| `seed` | Yes | `int` | Random seed for deterministic model initialization and parity input generation. |
| `input_shapes` | Yes | `dict[str, list[int]]` | Input names and concrete shapes used for export and runtime parity inputs. |
| `input_dtypes` | No | `dict[str, str]` | Optional per-input dtype map. Missing entries default to `float32`. |
| `tolerances` | Yes | `dict[str, float]` | Numerical comparison tolerances. Must contain `rtol` and `atol`. |
| `required_extras` | Yes | `list[str]` | Names of `pyproject.toml` optional-dependency groups required to materialize the model. |
| `export_kwargs` | No | `dict[str, Any]` | Extra keyword arguments for `torch.onnx.export()`. For transformer manifests, `config_class` and `model_class` are reserved builder hints and are removed before export. |

Invalid manifests fail fast.

- Missing files raise `FileNotFoundError`.
- Missing required fields or malformed values raise `ValueError`.

Example:

```yaml
family: transformers
model_name: Bert
opset: 17
pretrained: false
seed: 42
input_shapes:
  input_ids: [1, 128]
  attention_mask: [1, 128]
  token_type_ids: [1, 128]
input_dtypes:
  input_ids: int64
  attention_mask: int64
  token_type_ids: int64
tolerances:
  rtol: 1.0e-4
  atol: 1.0e-4
required_extras:
  - models
export_kwargs:
  config_class: BertConfig
  model_class: BertModel
```

## Execution Prerequisites

The current reference-model suite depends on both development tooling and model-export extras.

```bash
pip install -e ".[dev,models]"
```

- The `dev` extra provides `pytest` and `onnxruntime`, which are required to execute the parity assertions.
- The `models` extra provides `pyyaml`, `torchvision`, `transformers`, and `onnxscript`, which are required to
  parse manifests and materialize the current reference models.
- Current manifests declare `required_extras: [models]`, but test execution also assumes the `dev` toolchain.

All current model-family tests use the `model_validation` marker declared in `pyproject.toml`.

```bash
pytest tests/models/ -m model_validation -v
pytest tests/models/test_smoke.py -m model_validation -v
pytest tests/ -m "not model_validation" -v
```

When optional model-family dependencies are unavailable, the tests skip by catching `ImportError` during
materialization rather than failing with an import-time crash.

## Materialization and Cache Policy

`materialize(manifest, cache_root=None)` stores exported ONNX files in a cache directory and reuses them on
subsequent calls.

- The default cache root is `~/.cache/protofx/models`.
- Callers can override the root with the `cache_root` argument. The current tests pass `tmp_path` so each run
  uses an isolated temporary cache.

The current relative cache path is:

```text
<family>/<model_name>/opset<opset>/pretrained=<pretrained>_<export_kwargs_hash>.onnx
```

`export_kwargs_hash` is the first 8 hex characters of:

```text
sha256(json.dumps(export_kwargs, sort_keys=True, default=str))
```

Current cache identity behavior is therefore narrower than the full manifest schema.

| Manifest field(s) | Affect cache path? | Current behavior |
|-------------------|--------------------|------------------|
| `family`, `model_name`, `opset`, `pretrained`, `export_kwargs` | Yes | Changing any of these produces a different cache path. |
| `seed`, `input_shapes`, `input_dtypes`, `tolerances`, `required_extras` | No | Changing any of these does not invalidate an existing cache entry automatically. |

This is the concrete implementation contract contributors must account for today. If a non-keyed field changes
the exported artifact or parity expectations, use a fresh cache root or remove the stale cached file manually.

For transformer manifests, `config_class` and `model_class` live inside `export_kwargs`, so changing either
also changes cache identity.

## Local vs CI Execution

Reference-model validation remains a heavier suite than routine fast tests, and the current structure reflects
that distinction.

- The fastest local end-to-end check is `tests/models/test_smoke.py`, which exercises the full
  manifest-materialization, import, emit, and ORT parity path on a small `torchvision` model.
- Broader local investigation can run `tests/models/test_vision.py` or `tests/models/test_nlp.py` directly.
- Dedicated CI is expected to materialize assets and run the full marked suite rather than only the smoke test.

The `model_validation` marker is the primary switch separating these heavier checks from the default fast
feedback path.

## Relationship to Scripts

Helper tooling may exist for tasks such as downloading assets, exporting reference models, or prewarming a
cache. That tooling is useful, but it is not the architectural center of model validation.

ProtoFX treats the manifest-driven pytest suite as authoritative.
Scripts are support machinery; they are not the source of truth for coverage, pass/fail behavior, or reviewable
validation intent.
