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

`docs/dev/SUPPORT_MATRIX.md` may summarize the current exact model set for repository readers, but that visibility
surface is derived from manifests and pytest coverage rather than replacing them.

## Validation Layers

| Layer | Scope | Asset source | Expected scale |
|-------|-------|--------------|----------------|
| `tests/ops/` | Handler semantics and FX lowering behavior | Small hand-authored graphs and fixtures | Small |
| `tests/parity/` | Synthetic ONNX vs. ONNX Runtime numerical parity | Code-generated ONNX models | Small |
| `tests/models/` | Standard model-family validation | Manifest-declared, cache-backed reference assets | Large |

Downstream PyTorch-tooling compatibility is intentionally outside this document's scope.

- Checks for `torch.compile`, `torch.export`, FX quantization, and custom FX passes belong to
  `tests/downstream/` and `docs/dev/DOWNSTREAM_VALIDATION.md`.
- `tests/models/` remains focused on ONNX Runtime parity and manifest-backed end-to-end numerical
  validation, even when the same reference models are reused elsewhere.

## Repository Boundary

- Large reference ONNX binaries are not the normal git-tracked artifact for model-family validation.
- The repository keeps only human-reviewable declarations such as manifests, tolerances, and family coverage
  metadata.
- Small vendored ONNX fixtures remain acceptable only as narrow exceptions for bug regressions when
  code-generated or reproducibly exported alternatives are impractical.

This keeps source control focused on reviewable intent instead of generated binary payloads.

## Authoritative Sources

The exact current suite inventory may evolve, but the validation boundary does not.

- `tests/parity/` stays optimized for small code-generated models and fast operator-focused feedback.
- `tests/models/` carries heavier end-to-end validation concerns, optional dependencies, asset caching, and
  broader family-level coverage.
- `tests/models/manifests/` declares the exact current reference-model set.
- `tests/models/` owns the authoritative ONNX Runtime parity claims for those manifests.
- Downstream PyTorch-tooling compatibility claims remain outside this suite boundary even when they reuse
  the same manifests or exported artifacts.
- Helper scripts may assist materialization, but they do not replace the suite boundary above.
- `docs/dev/SUPPORT_MATRIX.md` may present a representative summary, but that summary does not own the underlying
  parity claim.

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
pytest tests/ -m "not model_validation" -v
```

Narrower file-scoped invocations are useful for local debugging, but the full marked suite remains the
authoritative validation surface.

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

## Execution Model

Reference-model validation is heavier than the default fast feedback path.

- The `model_validation` marker is the primary switch separating this suite from routine fast tests.
- Dedicated CI is expected to materialize assets and run the full marked suite.
- Local narrower runs are useful for debugging, but they do not redefine coverage or authority.

## Relationship to Scripts

Helper tooling may exist for tasks such as downloading assets, exporting reference models, or prewarming a
cache. That tooling is useful, but it is not the architectural center of model validation.

ProtoFX treats the manifest-driven pytest suite as authoritative.
Scripts are support machinery; they are not the source of truth for coverage, pass/fail behavior, or reviewable
validation intent.
