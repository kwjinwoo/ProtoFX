"""Shared fixtures and helpers for downstream PyTorch tooling validation tests."""

from __future__ import annotations

from collections.abc import Callable
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


def assert_export_roundtrip(
    model: ModelProto,
    inputs: dict[str, np.ndarray],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Assert eager vs ``torch.export`` round-trip numerical parity for a ProtoFX-emitted graph.

    Steps:
    1. Emit an eager ``GraphModule`` from the ONNX model.
    2. Run the eager graph with the supplied inputs.
    3. Export via ``torch.export.export`` and extract the module.
    4. Run the exported module with the same inputs.
    5. Assert the exported outputs are numerically close to the eager outputs.

    Args:
        model: A validated ``onnx.ModelProto``.
        inputs: Mapping from input name to numpy array.
        rtol: Relative tolerance for ``torch.testing.assert_close``.
        atol: Absolute tolerance for ``torch.testing.assert_close``.

    Raises:
        AssertionError: If exported outputs diverge from eager outputs.
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

    # Export round-trip
    exported = torch.export.export(gm, tuple(torch_inputs))
    exported_gm = exported.module()
    exported_outputs = exported_gm(*torch_inputs)
    if isinstance(exported_outputs, torch.Tensor):
        exported_outputs = (exported_outputs,)

    assert len(eager_outputs) == len(exported_outputs), (
        f"Output count mismatch: eager={len(eager_outputs)}, exported={len(exported_outputs)}"
    )

    for i, (eager_out, exported_out) in enumerate(zip(eager_outputs, exported_outputs, strict=True)):
        torch.testing.assert_close(
            exported_out,
            eager_out,
            rtol=rtol,
            atol=atol,
            msg=f"Output {i}: exported vs eager mismatch",
        )


def assert_quantize_survives_pt2e(
    model: ModelProto,
    inputs: dict[str, np.ndarray],
) -> None:
    """Assert that a ProtoFX-emitted graph survives the PT2E quantization pipeline.

    Steps:
    1. Emit an eager ``GraphModule`` from the ONNX model.
    2. Export via ``torch.export.export``.
    3. Apply ``prepare_pt2e`` with an ``XNNPACKQuantizer``.
    4. Calibrate the prepared model with the supplied inputs.
    5. Apply ``convert_pt2e`` to produce a quantized model.
    6. Execute the quantized model and verify output shapes match the eager model.

    Args:
        model: A validated ``onnx.ModelProto``.
        inputs: Mapping from input name to numpy array.

    Raises:
        AssertionError: If any step in the PT2E quantization pipeline fails or output shapes diverge.
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
            f"Output {i}: shape mismatch: quantized={quant_out.shape}, expected={exp_shape}"
        )


def assert_fx_pass_survives(
    model: ModelProto,
    inputs: dict[str, np.ndarray],
    pass_fn: Callable[[torch.fx.GraphModule, list[torch.Tensor]], torch.fx.GraphModule],
) -> None:
    """Assert that an FX pass executes successfully and output shapes are preserved.

    Steps:
    1. Emit an eager ``GraphModule`` from the ONNX model.
    2. Run the eager graph to capture reference output shapes.
    3. Apply ``pass_fn`` to the ``GraphModule``.
    4. Run the transformed ``GraphModule`` and verify shapes match the reference.

    Args:
        model: A validated ``onnx.ModelProto``.
        inputs: Mapping from input name to numpy array.
        pass_fn: A callable that takes a ``GraphModule`` and sample inputs, applies an FX
            transformation, and returns the (possibly mutated) ``GraphModule``.

    Raises:
        AssertionError: If the pass raises an exception or output shapes diverge.
    """
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

    # Apply FX pass
    transformed_gm = pass_fn(gm, torch_inputs)

    # Forward on the transformed graph
    transformed_outputs = transformed_gm(*torch_inputs)
    if isinstance(transformed_outputs, torch.Tensor):
        transformed_outputs = (transformed_outputs,)

    assert len(transformed_outputs) == len(expected_shapes), (
        f"Output count mismatch: transformed={len(transformed_outputs)}, expected={len(expected_shapes)}"
    )

    for i, (trans_out, exp_shape) in enumerate(zip(transformed_outputs, expected_shapes, strict=True)):
        assert trans_out.shape == exp_shape, (
            f"Output {i}: shape mismatch: transformed={trans_out.shape}, expected={exp_shape}"
        )
