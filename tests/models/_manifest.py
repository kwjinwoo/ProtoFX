"""YAML manifest schema for reference-model validation declarations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelManifest:
    """Parsed reference-model manifest.

    Attributes:
        family: Model family identifier (e.g. ``"torchvision"``, ``"transformers"``).
        model_name: Model constructor or identifier within the family.
        opset: ONNX opset version to use during export.
        pretrained: Whether to load pretrained weights during export.
        input_shapes: Mapping from input name to shape list (e.g. ``{"input": [1, 3, 224, 224]}``).
        tolerances: Numerical comparison tolerances with ``rtol`` and ``atol`` keys.
        export_kwargs: Extra keyword arguments forwarded to ``torch.onnx.export``.
    """

    family: str
    model_name: str
    opset: int
    pretrained: bool
    input_shapes: dict[str, list[int]]
    tolerances: dict[str, float]
    export_kwargs: dict[str, Any] = field(default_factory=dict)


_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {"family", "model_name", "opset", "pretrained", "input_shapes", "tolerances"}
)


def load_manifest(path: Path) -> ModelManifest:
    """Load and validate a YAML manifest file into a ``ModelManifest``.

    Args:
        path: Path to the YAML manifest file.

    Returns:
        A validated ``ModelManifest`` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If required fields are missing or have invalid types.
    """
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    if not isinstance(raw, dict):
        msg = f"Manifest must be a YAML mapping, got {type(raw).__name__}"
        raise ValueError(msg)

    missing = _REQUIRED_FIELDS - raw.keys()
    if missing:
        msg = f"Manifest is missing required fields: {sorted(missing)}"
        raise ValueError(msg)

    tolerances = raw["tolerances"]
    if not isinstance(tolerances, dict) or "rtol" not in tolerances or "atol" not in tolerances:
        msg = "tolerances must be a mapping with 'rtol' and 'atol' keys"
        raise ValueError(msg)

    return ModelManifest(
        family=str(raw["family"]),
        model_name=str(raw["model_name"]),
        opset=int(raw["opset"]),
        pretrained=bool(raw["pretrained"]),
        input_shapes={str(k): list(v) for k, v in raw["input_shapes"].items()},
        tolerances={str(k): float(v) for k, v in tolerances.items()},
        export_kwargs=raw.get("export_kwargs", {}),
    )
