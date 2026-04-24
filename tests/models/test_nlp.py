"""NLP model family parity tests (BERT, RoBERTa, DistilBERT, GPT2)."""

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


@pytest.mark.model_validation
class TestRoBERTaParity:
    """End-to-end ORT vs ProtoFX parity for RoBERTa."""

    def test_roberta_parity(self, tmp_path: Path) -> None:
        """RoBERTa manifest must produce numerically close ORT and ProtoFX outputs."""
        manifest = load_manifest(_MANIFESTS_DIR / "nlp" / "roberta.yaml")
        try:
            onnx_path = materialize(manifest, cache_root=tmp_path)
        except ImportError as exc:
            pytest.skip(f"Optional dependency unavailable: {exc}")
        assert_model_parity(onnx_path, manifest)


@pytest.mark.model_validation
class TestDistilBERTParity:
    """End-to-end ORT vs ProtoFX parity for DistilBERT."""

    def test_distilbert_parity(self, tmp_path: Path) -> None:
        """DistilBERT manifest must produce numerically close ORT and ProtoFX outputs."""
        manifest = load_manifest(_MANIFESTS_DIR / "nlp" / "distilbert.yaml")
        try:
            onnx_path = materialize(manifest, cache_root=tmp_path)
        except ImportError as exc:
            pytest.skip(f"Optional dependency unavailable: {exc}")
        assert_model_parity(onnx_path, manifest)


@pytest.mark.model_validation
class TestGPT2Parity:
    """End-to-end ORT vs ProtoFX parity for GPT2Model."""

    def test_gpt2_parity(self, tmp_path: Path) -> None:
        """GPT2 manifest must produce numerically close ORT and ProtoFX outputs."""
        manifest = load_manifest(_MANIFESTS_DIR / "nlp" / "gpt2.yaml")
        try:
            onnx_path = materialize(manifest, cache_root=tmp_path)
        except ImportError as exc:
            pytest.skip(f"Optional dependency unavailable: {exc}")
        assert_model_parity(onnx_path, manifest)
