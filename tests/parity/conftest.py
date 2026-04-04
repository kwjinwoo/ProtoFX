"""Parity test helpers: run ONNX models through ORT and ProtoFX, then compare outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
import torch

from protofx.emitters import emit_graph
from protofx.importers import import_model

if TYPE_CHECKING:
    from onnx import ModelProto


def run_ort(model: ModelProto, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Run an ONNX model through ONNX Runtime and return output arrays.

    Args:
        model: A validated ``onnx.ModelProto``.
        inputs: Mapping from input name to numpy array.

    Returns:
        List of numpy arrays, one per model output.
    """
    import onnxruntime as ort

    onnx.checker.check_model(model)
    session = ort.InferenceSession(model.SerializeToString())
    return session.run(None, inputs)


def run_protofx(model: ModelProto, inputs: dict[str, np.ndarray]) -> list[torch.Tensor]:
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


def assert_parity(
    model: ModelProto,
    inputs: dict[str, np.ndarray],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Assert numerical parity between ORT and ProtoFX outputs.

    Args:
        model: The ONNX model to compare.
        inputs: Mapping from input name to numpy array.
        rtol: Relative tolerance for ``np.allclose``.
        atol: Absolute tolerance for ``np.allclose``.

    Raises:
        AssertionError: If any output pair is not close within the given tolerance.
    """
    ort_outputs = run_ort(model, inputs)
    pfx_outputs = run_protofx(model, inputs)

    assert len(ort_outputs) == len(pfx_outputs), (
        f"Output count mismatch: ORT={len(ort_outputs)}, ProtoFX={len(pfx_outputs)}"
    )

    for i, (ort_out, pfx_out) in enumerate(zip(ort_outputs, pfx_outputs, strict=True)):
        pfx_np = pfx_out.detach().cpu().numpy()
        np.testing.assert_allclose(
            pfx_np,
            ort_out,
            rtol=rtol,
            atol=atol,
            err_msg=f"Output {i} mismatch",
        )
