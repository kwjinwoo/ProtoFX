"""Shared fixtures and helpers for downstream PyTorch tooling validation tests."""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING

import numpy as np
import torch

from protofx.emitters import emit_graph
from protofx.importers import import_model

if TYPE_CHECKING:
    from onnx import ModelProto


def build_eager_gm(model: ModelProto) -> torch.fx.GraphModule:
    """Import an ONNX model and emit an eager ``GraphModule``.

    Args:
        model: A validated ``onnx.ModelProto``.

    Returns:
        The emitted ``torch.fx.GraphModule``.
    """
    ir_graph = import_model(model)
    return emit_graph(ir_graph)


def assert_compile_parity(
    model: ModelProto,
    inputs: dict[str, np.ndarray],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Assert eager vs ``torch.compile`` numerical parity for a ProtoFX-emitted graph.

    Steps:
    1. Emit an eager ``GraphModule`` from the ONNX model.
    2. Run the eager graph with the supplied inputs.
    3. ``torch.compile`` the same ``GraphModule`` and run it.
    4. Assert the compiled outputs are numerically close to the eager outputs.

    Args:
        model: A validated ``onnx.ModelProto``.
        inputs: Mapping from input name to numpy array.
        rtol: Relative tolerance for ``torch.testing.assert_close``.
        atol: Absolute tolerance for ``torch.testing.assert_close``.

    Raises:
        AssertionError: If compiled outputs diverge from eager outputs.
    """
    gm = build_eager_gm(model)

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

    for i, (eager_out, compiled_out) in enumerate(zip(eager_outputs, compiled_outputs, strict=True)):
        torch.testing.assert_close(
            compiled_out,
            eager_out,
            rtol=rtol,
            atol=atol,
            msg=f"Output {i}: compiled vs eager mismatch",
        )


def _default_qconfig_backend() -> str:
    """Return the default quantization backend for the current platform.

    Returns:
        ``"qnnpack"`` on macOS/ARM, ``"x86"`` otherwise.
    """
    if platform.system() == "Darwin" or platform.machine() in ("arm64", "aarch64"):
        return "qnnpack"
    return "x86"


def assert_quantize_survives(
    model: ModelProto,
    inputs: dict[str, np.ndarray],
    *,
    qconfig_backend: str | None = None,
) -> None:
    """Assert that a ProtoFX-emitted graph survives the FX post-training quantization pipeline.

    Steps:
    1. Emit an eager ``GraphModule`` from the ONNX model.
    2. Run ``prepare_fx`` with a default static qconfig mapping.
    3. Calibrate the prepared model with the supplied inputs.
    4. Run ``convert_fx`` to produce a quantized model.
    5. Execute the quantized model and verify output shapes match the eager model.

    Args:
        model: A validated ``onnx.ModelProto``.
        inputs: Mapping from input name to numpy array.
        qconfig_backend: Quantization backend string (e.g. ``"x86"``, ``"qnnpack"``).
            Defaults to a platform-appropriate backend.

    Raises:
        AssertionError: If any step in the quantization pipeline fails or output shapes diverge.
    """
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    backend = qconfig_backend or _default_qconfig_backend()
    torch.backends.quantized.engine = backend

    gm = build_eager_gm(model)

    input_names = [
        inp.name for inp in model.graph.input if inp.name not in {init.name for init in model.graph.initializer}
    ]
    torch_inputs = [torch.from_numpy(inputs[name]) for name in input_names]

    # Eager forward — capture reference output shapes
    eager_outputs = gm(*torch_inputs)
    if isinstance(eager_outputs, torch.Tensor):
        eager_outputs = (eager_outputs,)
    expected_shapes = [out.shape for out in eager_outputs]

    # Prepare
    qconfig_mapping = get_default_qconfig_mapping(backend)
    prepared = prepare_fx(gm, qconfig_mapping, example_inputs=torch_inputs)

    # Calibrate
    prepared(*torch_inputs)

    # Convert
    quantized = convert_fx(prepared)

    # Execute quantized model
    quant_outputs = quantized(*torch_inputs)
    if isinstance(quant_outputs, torch.Tensor):
        quant_outputs = (quant_outputs,)

    assert len(quant_outputs) == len(expected_shapes), (
        f"Output count mismatch: quantized={len(quant_outputs)}, expected={len(expected_shapes)}"
    )

    for i, (quant_out, exp_shape) in enumerate(zip(quant_outputs, expected_shapes, strict=True)):
        assert quant_out.shape == exp_shape, (
            f"Output {i}: shape mismatch: quantized={quant_out.shape}, expected={exp_shape}"
        )
