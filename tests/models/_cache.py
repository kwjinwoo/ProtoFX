"""Cache-backed ONNX export manager for reference-model validation.

Exports PyTorch models (torchvision / transformers) to ONNX and stores them
in a configurable cache directory.  Cached artifacts are reused on subsequent
calls to avoid redundant exports.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tests.models._manifest import ModelManifest

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "protofx" / "models"


def _export_kwargs_hash(export_kwargs: dict) -> str:
    """Return an 8-character SHA-256 digest of serialized *export_kwargs*.

    Args:
        export_kwargs: The export keyword arguments to hash.

    Returns:
        An 8-character hex digest string.
    """
    serialized = json.dumps(export_kwargs, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:8]


def _cache_key(manifest: ModelManifest) -> Path:
    """Build a relative cache path from the manifest identity fields.

    Args:
        manifest: A validated model manifest.

    Returns:
        Relative ``Path`` incorporating family, model_name, opset, pretrained,
        and an export_kwargs content hash.
    """
    kwargs_hash = _export_kwargs_hash(manifest.export_kwargs)
    filename = f"pretrained={manifest.pretrained}_{kwargs_hash}.onnx"
    return Path(manifest.family) / manifest.model_name / f"opset{manifest.opset}" / filename


def _build_torchvision_model(manifest: ModelManifest) -> torch.nn.Module:
    """Instantiate a torchvision model from the manifest declaration.

    Args:
        manifest: Manifest with ``family == "torchvision"``.

    Returns:
        A ``torch.nn.Module`` in eval mode.

    Raises:
        ImportError: If ``torchvision`` is not installed.
        AttributeError: If *model_name* is not found in ``torchvision.models``.
    """
    import torchvision.models as tvm

    if not manifest.pretrained:
        torch.manual_seed(manifest.seed)
    factory = getattr(tvm, manifest.model_name)
    weights = "DEFAULT" if manifest.pretrained else None
    model: torch.nn.Module = factory(weights=weights)
    return model.eval()


def _build_transformers_model(manifest: ModelManifest) -> torch.nn.Module:
    """Instantiate a transformers model from the manifest declaration.

    Args:
        manifest: Manifest with ``family == "transformers"``.

    Raises:
        NotImplementedError: Always — transformers export is deferred to a future commit.
    """
    raise NotImplementedError("transformers model export is not yet implemented")


def _build_model(manifest: ModelManifest) -> torch.nn.Module:
    """Dispatch model construction to the appropriate family builder.

    Args:
        manifest: A validated model manifest.

    Returns:
        A ``torch.nn.Module`` in eval mode.

    Raises:
        ValueError: If the manifest *family* is not supported.
    """
    match manifest.family:
        case "torchvision":
            return _build_torchvision_model(manifest)
        case "transformers":
            return _build_transformers_model(manifest)
        case _:
            msg = f"Unsupported model family: {manifest.family!r}"
            raise ValueError(msg)


def _export_onnx(model: torch.nn.Module, manifest: ModelManifest, dest: Path) -> None:
    """Export a PyTorch model to ONNX and write it to *dest*.

    Args:
        model: The PyTorch module to export.
        manifest: Manifest supplying opset and input shape information.
        dest: Target file path for the exported ``.onnx`` model.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(manifest.seed)
    dummy_inputs: dict[str, torch.Tensor] = {}
    for name, shape in manifest.input_shapes.items():
        dummy_inputs[name] = torch.randn(*shape)

    input_names = list(manifest.input_shapes.keys())

    args = tuple(dummy_inputs[n] for n in input_names)

    torch.onnx.export(
        model,
        args,
        str(dest),
        opset_version=manifest.opset,
        input_names=input_names,
        **manifest.export_kwargs,
    )
    logger.info("Exported ONNX model to %s", dest)


def materialize(manifest: ModelManifest, cache_root: Path | None = None) -> Path:
    """Materialize an ONNX model from a manifest, using the cache when possible.

    If a cached export already exists at the expected path, it is returned
    immediately.  Otherwise the model is instantiated, exported to ONNX, and
    stored in the cache.

    Args:
        manifest: A validated model manifest.
        cache_root: Root directory for the cache.  Defaults to
            ``~/.cache/protofx/models``.

    Returns:
        Absolute ``Path`` to the cached ``.onnx`` file.
    """
    root = cache_root or _DEFAULT_CACHE_ROOT
    dest = root / _cache_key(manifest)

    if dest.exists():
        logger.debug("Cache hit: %s", dest)
        return dest

    logger.info("Cache miss — exporting %s/%s (opset %d)", manifest.family, manifest.model_name, manifest.opset)
    model = _build_model(manifest)
    _export_onnx(model, manifest, dest)
    return dest
