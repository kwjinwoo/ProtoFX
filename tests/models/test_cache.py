"""Tests for the torchvision/transformers export cache manager."""

from __future__ import annotations

from pathlib import Path

from tests.models._manifest import load_manifest

_MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"


class TestMaterialize:
    """Verify that materialize() exports an ONNX model and caches it."""

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
