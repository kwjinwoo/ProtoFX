"""Tests for the opset compatibility matrix generator."""

from __future__ import annotations

from protofx.ops._registry import list_registry


class TestGenerateOpsetMatrix:
    """Verify the opset compatibility matrix generator produces correct output."""

    def test_generate_returns_nonempty_string(self) -> None:
        """generate_opset_matrix must return a non-empty markdown string."""
        from scripts.gen_opset_matrix import generate_opset_matrix

        result = generate_opset_matrix()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_all_registered_ops(self) -> None:
        """The generated matrix must contain every registered op name."""
        from scripts.gen_opset_matrix import generate_opset_matrix

        result = generate_opset_matrix()
        registry = list_registry()
        for op_name in registry:
            if op_name.startswith("_"):
                continue  # Skip test-only registrations
            assert op_name in result, f"{op_name} not found in matrix"

    def test_contains_opset_columns(self) -> None:
        """The generated matrix must have columns for opsets 11 through 21."""
        from scripts.gen_opset_matrix import generate_opset_matrix

        result = generate_opset_matrix()
        for opset in range(11, 22):
            assert str(opset) in result, f"opset {opset} column not found"

    def test_supported_ops_show_check(self) -> None:
        """Ops within their declared opset range must show a check mark."""
        from scripts.gen_opset_matrix import generate_opset_matrix

        result = generate_opset_matrix()
        # Relu has range (11, 21) — the Relu row should contain check marks
        lines = result.split("\n")
        relu_lines = [line for line in lines if "| Relu " in line or "| Relu|" in line]
        assert len(relu_lines) == 1
        assert "\u2705" in relu_lines[0] or "✅" in relu_lines[0]

    def test_unsupported_ops_show_dash(self) -> None:
        """Ops outside their declared opset range must show a dash."""
        from scripts.gen_opset_matrix import generate_opset_matrix

        result = generate_opset_matrix()
        # Gelu has range (20, 21) — the Gelu row should have dashes for opsets 11-19
        lines = result.split("\n")
        gelu_lines = [line for line in lines if "| Gelu " in line or "| Gelu|" in line]
        assert len(gelu_lines) == 1
        assert "-" in gelu_lines[0]
