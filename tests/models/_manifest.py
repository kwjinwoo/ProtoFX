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
        seed: Random seed for deterministic model initialization and input generation.
        input_shapes: Mapping from input name to shape list (e.g. ``{"input": [1, 3, 224, 224]}``).
        input_dtypes: Optional mapping from input name to dtype string (e.g. ``{"input_ids": "int64"}``). Defaults
            to ``"float32"`` for inputs not listed.
        tolerances: Numerical comparison tolerances with ``rtol`` and ``atol`` keys.
        required_extras: ``pyproject.toml`` optional-dependency groups needed to materialize this model.
        export_kwargs: Extra keyword arguments forwarded to ``torch.onnx.export``.
    """

    family: str
    model_name: str
    opset: int
    pretrained: bool
    seed: int
    input_shapes: dict[str, list[int]]
    tolerances: dict[str, float]
    required_extras: list[str]
    input_dtypes: dict[str, str] = field(default_factory=dict)
    export_kwargs: dict[str, Any] = field(default_factory=dict)


_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {"family", "model_name", "opset", "pretrained", "seed", "input_shapes", "tolerances", "required_extras"}
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

    required_extras = raw["required_extras"]
    if not isinstance(required_extras, list) or not all(isinstance(e, str) for e in required_extras):
        msg = "required_extras must be a list of strings"
        raise ValueError(msg)

    input_dtypes_raw = raw.get("input_dtypes", {})
    if not isinstance(input_dtypes_raw, dict):
        msg = "input_dtypes must be a mapping from input name to dtype string"
        raise ValueError(msg)

    return ModelManifest(
        family=str(raw["family"]),
        model_name=str(raw["model_name"]),
        opset=int(raw["opset"]),
        pretrained=bool(raw["pretrained"]),
        seed=int(raw["seed"]),
        input_shapes={str(k): list(v) for k, v in raw["input_shapes"].items()},
        tolerances={str(k): float(v) for k, v in tolerances.items()},
        required_extras=[str(e) for e in required_extras],
        input_dtypes={str(k): str(v) for k, v in input_dtypes_raw.items()},
        export_kwargs=raw.get("export_kwargs", {}),
    )
