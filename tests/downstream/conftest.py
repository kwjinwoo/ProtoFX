"""Shared fixtures and helpers for downstream PyTorch tooling validation tests."""

from __future__ import annotations

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
