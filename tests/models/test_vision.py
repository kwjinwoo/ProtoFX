"""Vision model family parity tests (ResNet18, ResNet50, ViT-B/16, EfficientNet-B0, MobileNetV2, MobileNetV3-Small)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.models._cache import materialize
from tests.models._manifest import load_manifest
from tests.models.conftest import assert_model_parity

_MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"


@pytest.mark.model_validation
class TestResNet18Parity:
    """End-to-end ORT vs ProtoFX parity for ResNet18."""

    def test_resnet18_parity(self, tmp_path: Path) -> None:
        """ResNet18 manifest must produce numerically close ORT and ProtoFX outputs."""
        manifest = load_manifest(_MANIFESTS_DIR / "vision" / "resnet18.yaml")
        try:
            onnx_path = materialize(manifest, cache_root=tmp_path)
        except ImportError as exc:
            pytest.skip(f"Optional dependency unavailable: {exc}")
        assert_model_parity(onnx_path, manifest)


@pytest.mark.model_validation
class TestViTB16Parity:
    """End-to-end ORT vs ProtoFX parity for ViT-B/16."""

    def test_vit_b_16_parity(self, tmp_path: Path) -> None:
        """ViT-B/16 manifest must produce numerically close ORT and ProtoFX outputs."""
        manifest = load_manifest(_MANIFESTS_DIR / "vision" / "vit_b_16.yaml")
        try:
            onnx_path = materialize(manifest, cache_root=tmp_path)
        except ImportError as exc:
            pytest.skip(f"Optional dependency unavailable: {exc}")
        assert_model_parity(onnx_path, manifest)


@pytest.mark.model_validation
class TestResNet50Parity:
    """End-to-end ORT vs ProtoFX parity for ResNet50."""

    def test_resnet50_parity(self, tmp_path: Path) -> None:
        """ResNet50 manifest must produce numerically close ORT and ProtoFX outputs."""
        manifest = load_manifest(_MANIFESTS_DIR / "vision" / "resnet50.yaml")
        try:
            onnx_path = materialize(manifest, cache_root=tmp_path)
        except ImportError as exc:
            pytest.skip(f"Optional dependency unavailable: {exc}")
        assert_model_parity(onnx_path, manifest)


@pytest.mark.model_validation
class TestEfficientNetB0Parity:
    """End-to-end ORT vs ProtoFX parity for EfficientNet-B0."""

    def test_efficientnet_b0_parity(self, tmp_path: Path) -> None:
        """EfficientNet-B0 manifest must produce numerically close ORT and ProtoFX outputs."""
        manifest = load_manifest(_MANIFESTS_DIR / "vision" / "efficientnet_b0.yaml")
        try:
            onnx_path = materialize(manifest, cache_root=tmp_path)
        except ImportError as exc:
            pytest.skip(f"Optional dependency unavailable: {exc}")
        assert_model_parity(onnx_path, manifest)


@pytest.mark.model_validation
class TestMobileNetV2Parity:
    """End-to-end ORT vs ProtoFX parity for MobileNetV2."""

    def test_mobilenet_v2_parity(self, tmp_path: Path) -> None:
        """MobileNetV2 manifest must produce numerically close ORT and ProtoFX outputs."""
        manifest = load_manifest(_MANIFESTS_DIR / "vision" / "mobilenet_v2.yaml")
        try:
            onnx_path = materialize(manifest, cache_root=tmp_path)
        except ImportError as exc:
            pytest.skip(f"Optional dependency unavailable: {exc}")
        assert_model_parity(onnx_path, manifest)


@pytest.mark.model_validation
class TestMobileNetV3SmallParity:
    """End-to-end ORT vs ProtoFX parity for MobileNetV3-Small."""

    def test_mobilenet_v3_small_parity(self, tmp_path: Path) -> None:
        """MobileNetV3-Small manifest must produce numerically close ORT and ProtoFX outputs."""
        manifest = load_manifest(_MANIFESTS_DIR / "vision" / "mobilenet_v3_small.yaml")
        try:
            onnx_path = materialize(manifest, cache_root=tmp_path)
        except ImportError as exc:
            pytest.skip(f"Optional dependency unavailable: {exc}")
        assert_model_parity(onnx_path, manifest)
