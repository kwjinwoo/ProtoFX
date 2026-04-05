"""NLP model family parity tests (BERT)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.models._cache import materialize
from tests.models._manifest import load_manifest
from tests.models.conftest import assert_model_parity

_MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"


@pytest.mark.model_validation
class TestBERTParity:
    """End-to-end ORT vs ProtoFX parity for BERT."""

    def test_bert_parity(self, tmp_path: Path) -> None:
        """BERT manifest must produce numerically close ORT and ProtoFX outputs."""
        manifest = load_manifest(_MANIFESTS_DIR / "nlp" / "bert.yaml")
        try:
            onnx_path = materialize(manifest, cache_root=tmp_path)
        except ImportError as exc:
            pytest.skip(f"Optional dependency unavailable: {exc}")
        assert_model_parity(onnx_path, manifest)
