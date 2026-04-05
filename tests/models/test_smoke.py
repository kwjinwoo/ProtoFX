"""Smoke end-to-end parity test for the manifest-driven model validation infrastructure."""

from __future__ import annotations

from pathlib import Path

import pytest

_MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"


@pytest.mark.model_validation
class TestSmokeModelParity:
    """End-to-end ORT vs ProtoFX parity for a small torchvision model declared via manifest."""

    def test_squeezenet_smoke_parity(self, materialized_model: tuple[Path, dict]) -> None:
        """SqueezeNet smoke manifest must produce numerically close ORT and ProtoFX outputs."""
        from tests.models.conftest import assert_model_parity

        onnx_path, manifest_data = materialized_model
        assert_model_parity(onnx_path, manifest_data)
