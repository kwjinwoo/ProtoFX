"""Dry-run script to identify missing ops for CLIP model support.

Exports ``openai/clip-vit-base-patch32`` (via transformers ``CLIPModel``)
to ONNX and attempts a full ProtoFX import → emit cycle.  Any
``NotImplementedError`` raised by the op registry is caught and the
missing op name is collected.  The final report lists all unsupported ops.

Usage::

    python scripts/clip_dry_run.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import onnx
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _export_clip_onnx(dest: Path) -> None:
    """Export CLIPModel to ONNX at *dest*."""
    import transformers

    torch.manual_seed(42)
    config = transformers.CLIPConfig()
    model = transformers.CLIPModel(config).eval()

    dummy_inputs = {
        "input_ids": torch.ones(1, 77, dtype=torch.int64),
        "pixel_values": torch.randn(1, 3, 224, 224),
        "attention_mask": torch.ones(1, 77, dtype=torch.int64),
    }

    args = (dummy_inputs["input_ids"], dummy_inputs["pixel_values"], dummy_inputs["attention_mask"])

    exported = torch.export.export(model, args, strict=False)
    torch.onnx.export(
        exported,
        (),
        str(dest),
        dynamo=True,
        opset_version=17,
        input_names=["input_ids", "pixel_values", "attention_mask"],
    )
    logger.info("Exported CLIP ONNX model to %s", dest)


def _collect_missing_ops(onnx_path: Path) -> list[str]:
    """Import the ONNX model and collect op names that raise NotImplementedError."""
    from protofx.importers import import_model

    model = onnx.load(str(onnx_path))
    ir_graph = import_model(model)

    # Walk IR nodes and try dispatching each op.
    from protofx.ops._registry import dispatch_op

    opset_version = None
    for opset_import in model.opset_import:
        if opset_import.domain == "" or opset_import.domain == "ai.onnx":
            opset_version = opset_import.version
            break

    missing: list[str] = []
    seen: set[str] = set()
    for node in ir_graph.nodes:
        if node.op_type in seen:
            continue
        seen.add(node.op_type)
        try:
            dispatch_op(node.op_type, opset_version)
        except NotImplementedError:
            missing.append(node.op_type)

    return missing


def _collect_all_ops(onnx_path: Path) -> list[str]:
    """Return sorted unique op types used in the ONNX model."""
    model = onnx.load(str(onnx_path))
    ops: set[str] = set()
    for node in model.graph.node:
        ops.add(node.op_type)
    return sorted(ops)


def main() -> None:
    """Run the CLIP dry-run analysis."""
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "clip.onnx"
        logger.info("Exporting CLIP model to ONNX...")
        _export_clip_onnx(onnx_path)

        all_ops = _collect_all_ops(onnx_path)
        logger.info("All ONNX ops used by CLIP (%d): %s", len(all_ops), ", ".join(all_ops))

        missing = _collect_missing_ops(onnx_path)
        if missing:
            logger.warning("Missing op handlers (%d): %s", len(missing), ", ".join(missing))
            sys.exit(1)
        else:
            logger.info("All ops are supported! CLIP can be fully converted.")
            sys.exit(0)


if __name__ == "__main__":
    main()
