"""Tests for the YAML manifest schema loader."""

from __future__ import annotations

from pathlib import Path

import pytest

_MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"


class TestLoadManifest:
    """Verify that load_manifest parses YAML into a ModelManifest with all required fields."""

    def test_smoke_manifest_loads(self) -> None:
        """Smoke manifest must parse and expose all required fields."""
        from tests.models._manifest import ModelManifest, load_manifest

        manifest = load_manifest(_MANIFESTS_DIR / "smoke" / "smoke.yaml")

        assert isinstance(manifest, ModelManifest)
        assert manifest.family == "torchvision"
        assert isinstance(manifest.model_name, str) and manifest.model_name
        assert isinstance(manifest.opset, int) and manifest.opset > 0
        assert isinstance(manifest.input_shapes, dict) and len(manifest.input_shapes) > 0
        assert isinstance(manifest.tolerances, dict)
        assert "rtol" in manifest.tolerances
        assert "atol" in manifest.tolerances

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        """A manifest missing a required field must raise ValueError."""
        from tests.models._manifest import load_manifest

        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("family: torchvision\n")

        with pytest.raises((ValueError, KeyError)):
            load_manifest(bad_yaml)
