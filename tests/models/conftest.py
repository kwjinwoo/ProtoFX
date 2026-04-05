"""Shared fixtures and helpers for reference-model validation tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx
import pytest
import torch

from protofx.emitters import emit_graph
from protofx.importers import import_model
from tests.models._cache import materialize
from tests.models._manifest import ModelManifest, load_manifest

if TYPE_CHECKING:
    from onnx import ModelProto

_MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"


@pytest.fixture
def smoke_manifest() -> ModelManifest:
    """Load the smoke manifest (SqueezeNet, random weights)."""
    return load_manifest(_MANIFESTS_DIR / "smoke" / "smoke.yaml")


@pytest.fixture
def materialized_model(smoke_manifest: ModelManifest, tmp_path: Path) -> tuple[Path, ModelManifest]:
    """Export the smoke model to ONNX (cached in tmp_path) and return ``(onnx_path, manifest)``.

    Skips the test if the required optional dependency is not installed.
    """
    try:
        onnx_path = materialize(smoke_manifest, cache_root=tmp_path)
    except ImportError as exc:
        pytest.skip(f"Optional dependency unavailable: {exc}")
    return onnx_path, smoke_manifest


# ---------------------------------------------------------------------------
# Parity helpers (mirrors tests/parity/conftest.py for model-level scope)
# ---------------------------------------------------------------------------


def _run_ort(model: ModelProto, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Run an ONNX model through ONNX Runtime and return output arrays.

    Args:
        model: A validated ``onnx.ModelProto``.
        inputs: Mapping from input name to numpy array.

    Returns:
        List of numpy arrays, one per model output.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(model.SerializeToString())
    return session.run(None, inputs)


def _run_protofx(model: ModelProto, inputs: dict[str, np.ndarray]) -> list[torch.Tensor]:
    """Run an ONNX model through the full ProtoFX pipeline (import → emit → forward).

    Args:
        model: A validated ``onnx.ModelProto``.
        inputs: Mapping from input name to numpy array.

    Returns:
        List of output tensors from the ``GraphModule`` forward pass.
    """
    ir_graph = import_model(model)
    gm = emit_graph(ir_graph)

    input_names = [
        inp.name for inp in model.graph.input if inp.name not in {init.name for init in model.graph.initializer}
    ]
    torch_inputs = [torch.from_numpy(inputs[name]) for name in input_names]

    outputs = gm(*torch_inputs)
    if isinstance(outputs, torch.Tensor):
        return [outputs]
    return list(outputs)


def assert_model_parity(onnx_path: Path, manifest: ModelManifest) -> None:
    """Assert ORT vs ProtoFX numerical parity for a materialized ONNX model.

    Generates random inputs according to the manifest ``input_shapes`` and
    compares outputs within the declared ``tolerances``.

    Args:
        onnx_path: Path to the exported ``.onnx`` file.
        manifest: The ``ModelManifest`` that declared this model.

    Raises:
        AssertionError: If any output pair exceeds the declared tolerances.
    """
    model = onnx.load(str(onnx_path))

    rng = np.random.default_rng(manifest.seed)
    inputs: dict[str, np.ndarray] = {}
    for name, shape in manifest.input_shapes.items():
        inputs[name] = rng.standard_normal(shape).astype(np.float32)

    ort_outputs = _run_ort(model, inputs)
    pfx_outputs = _run_protofx(model, inputs)

    assert len(ort_outputs) == len(pfx_outputs), (
        f"Output count mismatch: ORT={len(ort_outputs)}, ProtoFX={len(pfx_outputs)}"
    )

    rtol = manifest.tolerances["rtol"]
    atol = manifest.tolerances["atol"]
    for i, (ort_out, pfx_out) in enumerate(zip(ort_outputs, pfx_outputs, strict=True)):
        pfx_np = pfx_out.detach().cpu().numpy()
        np.testing.assert_allclose(
            pfx_np,
            ort_out,
            rtol=rtol,
            atol=atol,
            err_msg=f"Output {i} mismatch for {manifest.family}/{manifest.model_name}",
        )
