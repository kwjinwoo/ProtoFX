"""PT2E quantization validation for manifest-backed reference models.

Verifies that emitted ``GraphModule`` objects for representative reference
models survive the torchao PT2E quantization pipeline
(``torch.export`` -> ``prepare_pt2e`` -> calibration -> ``convert_pt2e`` -> execute)
without exceptions and produce outputs with the expected shape.
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


def _assert_quantize_model_survives(onnx_path: Path, manifest: ModelManifest) -> None:
    """Assert that a materialized reference model survives the PT2E quantization pipeline.

    Args:
        onnx_path: Path to the exported ``.onnx`` file.
        manifest: The ``ModelManifest`` that declared this model.

    Raises:
        AssertionError: If the quantization pipeline fails or output shapes diverge.
    """
    from torchao.quantization.pt2e.observer import MinMaxObserver
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from torchao.quantization.pt2e.quantizer import (
        QuantizationAnnotation,
        QuantizationConfig,
        QuantizationSpec,
        Quantizer,
        get_input_act_qspec,
        get_output_act_qspec,
    )

    act_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
    )
    weight_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        is_dynamic=False,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
    )
    qconfig = QuantizationConfig(
        input_activation=act_spec,
        output_activation=act_spec,
        weight=weight_spec,
        bias=None,
    )

    class _SimpleStaticQuantizer(Quantizer):
        """Minimal quantizer that annotates all call_function nodes with symmetric int8."""

        def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    input_qspec_map = {
                        arg: get_input_act_qspec(qconfig) for arg in node.args if isinstance(arg, torch.fx.Node)
                    }
                    node.meta["quantization_annotation"] = QuantizationAnnotation(
                        input_qspec_map=input_qspec_map,
                        output_qspec=get_output_act_qspec(qconfig),
                    )
            return gm

        def validate(self, gm: torch.fx.GraphModule) -> None:
            pass

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

    # Eager forward — capture reference output shapes
    eager_outputs = gm(*torch_inputs)
    if isinstance(eager_outputs, torch.Tensor):
        eager_outputs = (eager_outputs,)
    expected_shapes = [out.shape for out in eager_outputs]

    # Export and extract GraphModule
    exported = torch.export.export(gm, tuple(torch_inputs))
    exported_gm = exported.module()

    # Prepare with SimpleStaticQuantizer
    quantizer = _SimpleStaticQuantizer()
    prepared = prepare_pt2e(exported_gm, quantizer)

    # Calibrate
    prepared(*torch_inputs)

    # Convert
    quantized = convert_pt2e(prepared)

    # Execute quantized model
    quant_outputs = quantized(*torch_inputs)
    if isinstance(quant_outputs, torch.Tensor):
        quant_outputs = (quant_outputs,)

    assert len(quant_outputs) == len(expected_shapes), (
        f"Output count mismatch: quantized={len(quant_outputs)}, expected={len(expected_shapes)}"
    )

    for i, (quant_out, exp_shape) in enumerate(zip(quant_outputs, expected_shapes, strict=True)):
        assert quant_out.shape == exp_shape, (
            f"Output {i}: shape mismatch for {manifest.family}/{manifest.model_name}: "
            f"quantized={quant_out.shape}, expected={exp_shape}"
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
class TestQuantizeSqueezeNet:
    """FX quantization survival for SqueezeNet (smoke manifest)."""

    def test_squeezenet_quantize_survives(self, tmp_path: Path) -> None:
        """SqueezeNet must survive the FX quantization pipeline."""
        manifest = load_manifest(_MANIFESTS_DIR / "smoke" / "smoke.yaml")
        onnx_path = _materialize_or_skip(manifest, tmp_path)
        _assert_quantize_model_survives(onnx_path, manifest)
