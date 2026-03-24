"""Failing tests for the op handler registry."""

from __future__ import annotations

import pytest

from protofx.ops import dispatch_op, register_op


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
