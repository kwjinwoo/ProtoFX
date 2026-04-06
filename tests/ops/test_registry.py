"""Tests for the op handler registry."""

from __future__ import annotations

import pytest

from protofx.ops import dispatch_op, register_op
from protofx.ops._registry import list_registry


class TestRegisterOp:
    """Verify that @register_op registers handlers correctly."""

    def test_register_and_dispatch(self) -> None:
        """A handler registered with @register_op must be retrievable via dispatch_op."""

        @register_op("_TestOp")
        def _test_handler(node, args, fx_graph, module):
            return []

        handler = dispatch_op("_TestOp")
        assert handler is _test_handler

    def test_dispatch_unregistered_raises(self) -> None:
        """dispatch_op for an unregistered op must raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="__NoSuchOp__"):
            dispatch_op("__NoSuchOp__")

    def test_duplicate_register_raises(self) -> None:
        """Registering the same op name twice must raise ValueError."""

        @register_op("_DupOp")
        def _first(node, args, fx_graph, module):
            return []

        with pytest.raises(ValueError, match="_DupOp"):

            @register_op("_DupOp")
            def _second(node, args, fx_graph, module):
                return []


class TestOpsetRange:
    """Verify opset_range metadata storage and version enforcement."""

    def test_register_with_opset_range(self) -> None:
        """register_op with opset_range stores the range in the registry."""

        @register_op("_RangedOp", opset_range=(11, 17))
        def _handler(node, args, fx_graph, module):
            return []

        registry = list_registry()
        assert "_RangedOp" in registry
        assert registry["_RangedOp"] == (11, 17)

    def test_register_without_opset_range(self) -> None:
        """register_op without opset_range stores None."""

        @register_op("_NoRangeOp")
        def _handler(node, args, fx_graph, module):
            return []

        registry = list_registry()
        assert "_NoRangeOp" in registry
        assert registry["_NoRangeOp"] is None

    def test_dispatch_within_range_succeeds(self) -> None:
        """dispatch_op with opset_version inside the handler's range returns the handler."""

        @register_op("_InRangeOp", opset_range=(13, 21))
        def _handler(node, args, fx_graph, module):
            return []

        handler = dispatch_op("_InRangeOp", opset_version=17)
        assert handler is _handler

    def test_dispatch_at_range_boundaries(self) -> None:
        """dispatch_op at exact lower and upper bounds returns the handler."""

        @register_op("_BoundaryOp", opset_range=(11, 21))
        def _handler(node, args, fx_graph, module):
            return []

        assert dispatch_op("_BoundaryOp", opset_version=11) is _handler
        assert dispatch_op("_BoundaryOp", opset_version=21) is _handler

    def test_dispatch_below_range_raises(self) -> None:
        """dispatch_op with opset_version below the range raises NotImplementedError."""

        @register_op("_BelowRangeOp", opset_range=(13, 21))
        def _handler(node, args, fx_graph, module):
            return []

        with pytest.raises(NotImplementedError, match="opset version 10.*_BelowRangeOp.*13.*21"):
            dispatch_op("_BelowRangeOp", opset_version=10)

    def test_dispatch_above_range_raises(self) -> None:
        """dispatch_op with opset_version above the range raises NotImplementedError."""

        @register_op("_AboveRangeOp", opset_range=(11, 17))
        def _handler(node, args, fx_graph, module):
            return []

        with pytest.raises(NotImplementedError, match="opset version 20.*_AboveRangeOp.*11.*17"):
            dispatch_op("_AboveRangeOp", opset_version=20)

    def test_dispatch_none_version_skips_check(self) -> None:
        """dispatch_op with opset_version=None skips the version check."""

        @register_op("_NoneVersionOp", opset_range=(13, 21))
        def _handler(node, args, fx_graph, module):
            return []

        handler = dispatch_op("_NoneVersionOp", opset_version=None)
        assert handler is _handler

    def test_dispatch_no_range_ignores_version(self) -> None:
        """dispatch_op for a handler without opset_range ignores the opset_version arg."""

        @register_op("_NoRangeDispatchOp")
        def _handler(node, args, fx_graph, module):
            return []

        handler = dispatch_op("_NoRangeDispatchOp", opset_version=999)
        assert handler is _handler

    def test_list_registry_includes_all_ops(self) -> None:
        """list_registry returns all registered ops including built-in handlers."""
        registry = list_registry()
        # At minimum, the built-in ops from ops/ modules should be present
        assert "Relu" in registry
        assert "Add" in registry
        assert "Conv" in registry
