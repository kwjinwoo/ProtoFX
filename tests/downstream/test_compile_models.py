"""torch.compile validation for manifest-backed reference models.

Verifies that emitted ``GraphModule`` objects for representative reference
models survive ``torch.compile`` with the default inductor backend and
produce numerically close outputs compared to eager execution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from protofx.emitters import emit_graph
from protofx.importers import import_model
from tests.models._cache import materialize
from tests.models._manifest import ModelManifest, load_manifest

_MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "models" / "manifests"


def _assert_compile_model_parity(onnx_path: Path, manifest: ModelManifest) -> None:
    """Assert eager vs compiled parity for a materialized reference model.

    Args:
        onnx_path: Path to the exported ``.onnx`` file.
        manifest: The ``ModelManifest`` that declared this model.

    Raises:
        AssertionError: If compiled outputs diverge from eager outputs.
    """
    model = onnx.load(str(onnx_path))
    ir_graph = import_model(model)
    gm = emit_graph(ir_graph)

    rng = np.random.default_rng(manifest.seed)
    inputs: dict[str, np.ndarray] = {}
    for name, shape in manifest.input_shapes.items():
        dtype_str = manifest.input_dtypes.get(name, "float32")
        np_dtype = np.dtype(dtype_str)
        if np.issubdtype(np_dtype, np.floating):
            inputs[name] = rng.standard_normal(shape).astype(np_dtype)
        else:
            inputs[name] = np.ones(shape, dtype=np_dtype)

    input_names = [
        inp.name for inp in model.graph.input if inp.name not in {init.name for init in model.graph.initializer}
    ]
    torch_inputs = [torch.from_numpy(inputs[name]) for name in input_names]

    # Eager forward
    eager_outputs = gm(*torch_inputs)
    if isinstance(eager_outputs, torch.Tensor):
        eager_outputs = (eager_outputs,)

    # Compiled forward
    compiled_gm = torch.compile(gm, backend="inductor")
    compiled_outputs = compiled_gm(*torch_inputs)
    if isinstance(compiled_outputs, torch.Tensor):
        compiled_outputs = (compiled_outputs,)

    assert len(eager_outputs) == len(compiled_outputs), (
        f"Output count mismatch: eager={len(eager_outputs)}, compiled={len(compiled_outputs)}"
    )

    rtol = manifest.tolerances["rtol"]
    atol = manifest.tolerances["atol"]
    for i, (eager_out, compiled_out) in enumerate(zip(eager_outputs, compiled_outputs, strict=True)):
        torch.testing.assert_close(
            compiled_out,
            eager_out,
            rtol=rtol,
            atol=atol,
            msg=f"Output {i}: compiled vs eager mismatch for {manifest.family}/{manifest.model_name}",
        )


def _materialize_or_skip(manifest: ModelManifest, tmp_path: Path) -> Path:
    """Materialize a manifest model, skipping the test if dependencies are unavailable.

    Args:
        manifest: A loaded ``ModelManifest``.
        tmp_path: Temporary directory for ONNX cache.

    Returns:
        Path to the exported ``.onnx`` file.
    """
    try:
        return materialize(manifest, cache_root=tmp_path)
    except ImportError as exc:
        pytest.skip(f"Optional dependency unavailable: {exc}")


@pytest.mark.downstream_validation
@pytest.mark.model_validation
class TestCompileSqueezeNet:
    """torch.compile parity for SqueezeNet (smoke manifest)."""

    def test_squeezenet_compile_parity(self, tmp_path: Path) -> None:
        """Compiled SqueezeNet must match eager output."""
        manifest = load_manifest(_MANIFESTS_DIR / "smoke" / "smoke.yaml")
        onnx_path = _materialize_or_skip(manifest, tmp_path)
        _assert_compile_model_parity(onnx_path, manifest)


@pytest.mark.downstream_validation
@pytest.mark.model_validation
class TestCompileResNet18:
    """torch.compile parity for ResNet18."""

    def test_resnet18_compile_parity(self, tmp_path: Path) -> None:
        """Compiled ResNet18 must match eager output."""
        manifest = load_manifest(_MANIFESTS_DIR / "vision" / "resnet18.yaml")
        onnx_path = _materialize_or_skip(manifest, tmp_path)
        _assert_compile_model_parity(onnx_path, manifest)


@pytest.mark.downstream_validation
@pytest.mark.model_validation
class TestCompileBERT:
    """torch.compile parity for BERT."""

    def test_bert_compile_parity(self, tmp_path: Path) -> None:
        """Compiled BERT must match eager output."""
        manifest = load_manifest(_MANIFESTS_DIR / "nlp" / "bert.yaml")
        onnx_path = _materialize_or_skip(manifest, tmp_path)
        _assert_compile_model_parity(onnx_path, manifest)
