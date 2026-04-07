"""Tests for the torchvision/transformers export cache manager."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import pytest

from tests.models._manifest import load_manifest

_MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"


class TestMaterialize:
    """Verify that materialize() exports an ONNX model and caches it."""

    @pytest.fixture(autouse=True)
    def _require_torchvision(self) -> None:
        pytest.importorskip("torchvision")

    def test_materialize_returns_existing_onnx_path(self, tmp_path: Path) -> None:
        """materialize() must return a Path to an existing .onnx file."""
        from tests.models._cache import materialize

        manifest = load_manifest(_MANIFESTS_DIR / "smoke" / "smoke.yaml")
        onnx_path = materialize(manifest, cache_root=tmp_path)

        assert onnx_path.exists()
        assert onnx_path.suffix == ".onnx"

    def test_materialize_caches_on_second_call(self, tmp_path: Path) -> None:
        """A second materialize() call must reuse the cached file, not re-export."""
        from tests.models._cache import materialize

        manifest = load_manifest(_MANIFESTS_DIR / "smoke" / "smoke.yaml")
        path1 = materialize(manifest, cache_root=tmp_path)
        mtime1 = path1.stat().st_mtime

        path2 = materialize(manifest, cache_root=tmp_path)

        assert path1 == path2
        assert path2.stat().st_mtime == mtime1, "File was re-exported instead of reused from cache"


class TestCacheKeyIsolation:
    """Verify that cache keys differ when export-affecting fields differ."""

    def test_different_export_kwargs_produce_different_cache_paths(self) -> None:
        """Manifests that differ only in export_kwargs must not share a cache path."""
        from tests.models._cache import _cache_key

        base = load_manifest(_MANIFESTS_DIR / "smoke" / "smoke.yaml")
        variant = replace(base, export_kwargs={"dynamic_axes": {"input": {0: "batch"}}})

        assert _cache_key(base) != _cache_key(variant)


class TestReproducibleExport:
    """Verify that export from the same manifest produces numerically identical results."""

    @pytest.fixture(autouse=True)
    def _require_torchvision(self) -> None:
        pytest.importorskip("torchvision")

    def test_two_cold_exports_produce_same_ort_outputs(self, tmp_path: Path) -> None:
        """Two independent exports of the same manifest must yield identical ORT outputs."""
        import onnxruntime as ort

        from tests.models._cache import materialize

        manifest = load_manifest(_MANIFESTS_DIR / "smoke" / "smoke.yaml")

        cache_a = tmp_path / "cache_a"
        cache_b = tmp_path / "cache_b"
        path_a = materialize(manifest, cache_root=cache_a)
        path_b = materialize(manifest, cache_root=cache_b)

        rng = np.random.default_rng(manifest.seed)
        dummy_input = rng.standard_normal(manifest.input_shapes["input"]).astype(np.float32)
        feeds = {"input": dummy_input}

        model_a = onnx.load(str(path_a))
        sess_a = ort.InferenceSession(model_a.SerializeToString())
        out_a = sess_a.run(None, feeds)

        model_b = onnx.load(str(path_b))
        sess_b = ort.InferenceSession(model_b.SerializeToString())
        out_b = sess_b.run(None, feeds)

        for i, (a, b) in enumerate(zip(out_a, out_b, strict=True)):
            np.testing.assert_allclose(a, b, rtol=0, atol=0, err_msg=f"Output {i} differs between two cold exports")
