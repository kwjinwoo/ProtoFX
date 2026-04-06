"""Tests for the emitter package import and emit_graph behavior."""

from __future__ import annotations

import pytest
import torch
import torch.fx

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType


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


class TestOpsetEnforcement:
    """Verify that emit_graph enforces opset version constraints during dispatch."""

    def test_emit_graph_raises_on_unsupported_opset(self) -> None:
        """emit_graph must raise NotImplementedError when a node's opset_version is outside the handler's range."""
        from protofx.ops._registry import register_op

        @register_op("_EmitterTestOp", opset_range=(13, 17))
        def _handler(node, args, fx_graph, module):
            import torch

            return [fx_graph.call_function(torch.relu, args=(args[0],))]

        g = Graph(name="opset_test")
        x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)), name="X")
        node = g.make_node(
            op_type="_EmitterTestOp",
            inputs=[x],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(2, 3))],
            output_names=["Y"],
            opset_version=20,
        )
        g.set_graph_outputs(list(node.outputs))

        with pytest.raises(NotImplementedError, match="opset version 20.*_EmitterTestOp.*13.*17"):
            emit_graph(g)
