"""Multi-modal model family parity tests (CLIP)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.models._cache import materialize
from tests.models._manifest import load_manifest
from tests.models.conftest import assert_model_parity

_MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"


@pytest.mark.model_validation
class TestCLIPParity:
    """End-to-end ORT vs ProtoFX parity for CLIP."""

    def test_clip_parity(self, tmp_path: Path) -> None:
        """CLIP manifest must produce numerically close ORT and ProtoFX outputs."""
        manifest = load_manifest(_MANIFESTS_DIR / "multimodal" / "clip.yaml")
        try:
            onnx_path = materialize(manifest, cache_root=tmp_path)
        except ImportError as exc:
            pytest.skip(f"Optional dependency unavailable: {exc}")
        assert_model_parity(onnx_path, manifest)
