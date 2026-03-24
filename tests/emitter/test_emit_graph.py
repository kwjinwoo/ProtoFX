"""Failing tests for the emitter package import and emit_graph stub."""

from __future__ import annotations

import torch
import torch.fx

from protofx.emitters import emit_graph
from protofx.ir import Graph


class TestEmitGraphImport:
    """Verify that emit_graph is importable and returns a GraphModule."""

    def test_emit_graph_is_callable(self) -> None:
        """emit_graph must be importable and callable."""
        assert callable(emit_graph)

    def test_emit_graph_returns_graph_module(self) -> None:
        """emit_graph on an empty graph must return a torch.fx.GraphModule."""
        g = Graph(name="empty")
        g.set_graph_outputs([])

        result = emit_graph(g)

        assert isinstance(result, torch.fx.GraphModule)
