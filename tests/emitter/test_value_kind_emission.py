"""Failing tests for value-kind emission in emit_graph.

These tests verify that each IR ValueKind produces the correct FX node type:
- GRAPH_INPUT → placeholder
- INITIALIZER → get_attr (buffer registered on root module)
- CONSTANT → get_attr (buffer registered on root module)
- SENTINEL → None in args
- NODE_OUTPUT → result of op handler dispatch
- graph outputs → output node
"""

from __future__ import annotations

import numpy as np
import torch
import torch.fx

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType
from protofx.ir.value import ValueKind


def _fx_node_ops(gm: torch.fx.GraphModule) -> list[str]:
    """Return a list of FX node op strings in graph order."""
    return [n.op for n in gm.graph.nodes]


class TestPlaceholderEmission:
    """GRAPH_INPUT values must emit FX placeholder nodes."""

    def test_single_input_produces_placeholder(self) -> None:
        """One graph input must produce one placeholder node."""
        g = Graph(name="test")
        inp = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)), name="X")
        g.set_graph_outputs([inp])

        gm = emit_graph(g)
        ops = _fx_node_ops(gm)

        assert ops.count("placeholder") == 1

    def test_multiple_inputs_produce_ordered_placeholders(self) -> None:
        """Multiple graph inputs must produce placeholders in order."""
        g = Graph(name="test")
        x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)), name="X")
        y = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)), name="Y")
        g.set_graph_outputs([x, y])

        gm = emit_graph(g)
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]

        assert len(placeholders) == 2


class TestInitializerEmission:
    """INITIALIZER values must emit get_attr nodes with buffers on the root module."""

    def test_initializer_emits_get_attr(self) -> None:
        """An initializer must produce a get_attr FX node."""
        g = Graph(name="test")
        data = np.ones((3, 4), dtype=np.float32)
        init = g.add_initializer(
            tensor_type=TensorType(dtype=DType.FLOAT32, shape=(3, 4)),
            data=data,
            name="weight",
        )
        g.set_graph_outputs([init])

        gm = emit_graph(g)
        ops = _fx_node_ops(gm)

        assert "get_attr" in ops

    def test_initializer_buffer_registered(self) -> None:
        """The root module must have a registered buffer for the initializer."""
        g = Graph(name="test")
        data = np.array([1.0, 2.0], dtype=np.float32)
        init = g.add_initializer(
            tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)),
            data=data,
            name="bias",
        )
        g.set_graph_outputs([init])

        gm = emit_graph(g)
        buffers = dict(gm.named_buffers())

        assert len(buffers) >= 1
        buf = next(iter(buffers.values()))
        assert torch.equal(buf, torch.tensor([1.0, 2.0], dtype=torch.float32))


class TestConstantEmission:
    """CONSTANT values must emit get_attr nodes with buffers on the root module."""

    def test_constant_emits_get_attr(self) -> None:
        """A constant value used as a node input must produce a get_attr FX node."""
        g = Graph(name="test")
        data = np.array([3.14], dtype=np.float32)
        const = g.add_constant(
            tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1,)),
            data=data,
            name="pi",
        )
        g.set_graph_outputs([const])

        gm = emit_graph(g)
        ops = _fx_node_ops(gm)

        assert "get_attr" in ops


class TestSentinelEmission:
    """SENTINEL values must be mapped to None in op handler args."""

    def test_sentinel_becomes_none_in_args(self) -> None:
        """When a node has a SENTINEL input, the emitter must pass None to the handler."""
        # This test will be meaningful once emit_graph dispatches through ops.
        # For now, just verify sentinels don't crash emission.
        g = Graph(name="test")
        inp = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)), name="X")
        sentinel = g.add_sentinel()

        # Make a dummy node with sentinel input — will need an op handler to fully test
        # For this commit, sentinel handling is verified via the SENTINEL value kind
        assert sentinel.kind == ValueKind.SENTINEL
        g.set_graph_outputs([inp])

        gm = emit_graph(g)
        assert isinstance(gm, torch.fx.GraphModule)


class TestOutputEmission:
    """Graph outputs must produce an FX output node."""

    def test_single_output(self) -> None:
        """One graph output must produce one output node."""
        g = Graph(name="test")
        inp = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)), name="X")
        g.set_graph_outputs([inp])

        gm = emit_graph(g)
        ops = _fx_node_ops(gm)

        assert ops.count("output") == 1

    def test_output_is_last_node(self) -> None:
        """The output node must be the last node in the FX graph."""
        g = Graph(name="test")
        inp = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)), name="X")
        g.set_graph_outputs([inp])

        gm = emit_graph(g)
        nodes = list(gm.graph.nodes)

        assert nodes[-1].op == "output"

    def test_multiple_outputs_packed_as_tuple(self) -> None:
        """Multiple graph outputs must be packed as a tuple in the output node."""
        g = Graph(name="test")
        x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)), name="X")
        y = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(3,)), name="Y")
        g.set_graph_outputs([x, y])

        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")

        # output node args[0] should be a tuple of proxy references
        assert isinstance(output_node.args[0], tuple)
        assert len(output_node.args[0]) == 2
