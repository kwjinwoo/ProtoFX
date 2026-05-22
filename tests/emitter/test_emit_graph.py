"""Tests for the emitter package import and emit_graph behavior."""

from __future__ import annotations

import operator

import pytest
import torch
import torch.fx

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType
from protofx.ir.derived_shape import set_derived_shape


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


class TestChildGraphEmitterHelpers:
    """Verify internal child-graph emission helper behavior through temporary test ops."""

    def test_child_graph_callable_executes(self) -> None:
        """A handler can lower a child graph into a callable and execute it via FX."""
        from protofx.ops._registry import register_op

        @register_op("_EmitterChildGraphIdentity")
        def _handler(node, args, fx_graph, module):
            helper = getattr(module, "_protofx_child_graph_emitter", None)
            if helper is None:
                raise ValueError("missing internal child graph emitter helper")

            child_graph = node.subgraphs.get("then_branch")
            if not isinstance(child_graph, Graph):
                raise ValueError("missing child graph for _EmitterChildGraphIdentity")

            branch, _ = helper.make_callable(owner_node=node, branch_name="then_branch", child_graph=child_graph)
            packed = fx_graph.call_function(branch, args=(args[0],))
            return [fx_graph.call_function(operator.getitem, args=(packed, 0))]

        parent = Graph(name="parent")
        x = parent.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)), name="x")
        child = Graph(name="child", parent=parent)
        child_x = child.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)), name="x")
        child_identity = child.make_node(
            op_type="Identity",
            inputs=[child_x],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(2,))],
        )
        child.set_graph_outputs([child_identity.outputs[0]])

        node = parent.make_node(
            op_type="_EmitterChildGraphIdentity",
            inputs=[x],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(2,))],
            subgraphs={"then_branch": child},
            name="cf",
        )
        parent.set_graph_outputs([node.outputs[0]])
        gm = emit_graph(parent)

        (result,) = gm(torch.tensor([1.0, -2.0]))
        torch.testing.assert_close(result, torch.tensor([1.0, -2.0]))

    def test_child_graph_attr_names_avoid_collisions(self) -> None:
        """Lowering two child graphs with the same base name must produce unique module attributes."""
        from protofx.ops._registry import register_op

        @register_op("_EmitterChildGraphCollision")
        def _handler(node, args, fx_graph, module):
            helper = getattr(module, "_protofx_child_graph_emitter", None)
            if helper is None:
                raise ValueError("missing internal child graph emitter helper")

            child_graph = node.subgraphs.get("then_branch")
            if not isinstance(child_graph, Graph):
                raise ValueError("missing child graph for _EmitterChildGraphCollision")

            branch, _ = helper.make_callable(owner_node=node, branch_name="then_branch", child_graph=child_graph)
            packed = fx_graph.call_function(branch, args=(args[0],))
            return [fx_graph.call_function(operator.getitem, args=(packed, 0))]

        parent = Graph(name="parent")
        x = parent.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)), name="x")

        def _make_child() -> Graph:
            child = Graph(name="child", parent=parent)
            child_x = child.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)), name="x")
            child_identity = child.make_node(
                op_type="Identity",
                inputs=[child_x],
                output_types=[TensorType(dtype=DType.FLOAT32, shape=(2,))],
            )
            child.set_graph_outputs([child_identity.outputs[0]])
            return child

        n1 = parent.make_node(
            op_type="_EmitterChildGraphCollision",
            inputs=[x],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(2,))],
            subgraphs={"then_branch": _make_child()},
            name="same",
        )
        n2 = parent.make_node(
            op_type="_EmitterChildGraphCollision",
            inputs=[n1.outputs[0]],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(2,))],
            subgraphs={"then_branch": _make_child()},
            name="same",
        )
        parent.set_graph_outputs([n2.outputs[0]])
        gm = emit_graph(parent)

        (result,) = gm(torch.tensor([1.0, -2.0]))
        torch.testing.assert_close(result, torch.tensor([1.0, -2.0]))

    def test_child_graph_output_tuple_unpacking(self) -> None:
        """Handlers can unpack tuple outputs from child-graph callables."""
        from protofx.ops._registry import register_op

        @register_op("_EmitterChildGraphTuple")
        def _handler(node, args, fx_graph, module):
            helper = getattr(module, "_protofx_child_graph_emitter", None)
            if helper is None:
                raise ValueError("missing internal child graph emitter helper")

            child_graph = node.subgraphs.get("then_branch")
            if not isinstance(child_graph, Graph):
                raise ValueError("missing child graph for _EmitterChildGraphTuple")

            branch, arity = helper.make_callable(owner_node=node, branch_name="then_branch", child_graph=child_graph)
            packed = fx_graph.call_function(branch, args=(args[0],))
            if arity == 1:
                return [packed]
            return [fx_graph.call_function(operator.getitem, args=(packed, idx)) for idx in range(arity)]

        parent = Graph(name="parent")
        x = parent.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)), name="x")
        child = Graph(name="child", parent=parent)
        child_x = child.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)), name="x")
        out_a = child.make_node(
            op_type="Identity",
            inputs=[child_x],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(2,))],
        ).outputs[0]
        out_b = child.make_node(
            op_type="Neg",
            inputs=[child_x],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(2,))],
        ).outputs[0]
        child.set_graph_outputs([out_a, out_b])

        node = parent.make_node(
            op_type="_EmitterChildGraphTuple",
            inputs=[x],
            output_types=[
                TensorType(dtype=DType.FLOAT32, shape=(2,)),
                TensorType(dtype=DType.FLOAT32, shape=(2,)),
            ],
            subgraphs={"then_branch": child},
            name="tuple_cf",
        )
        parent.set_graph_outputs([node.outputs[0], node.outputs[1]])
        gm = emit_graph(parent)

        out_a_t, out_b_t = gm(torch.tensor([1.0, -2.0]))
        torch.testing.assert_close(out_a_t, torch.tensor([1.0, -2.0]))
        torch.testing.assert_close(out_b_t, torch.tensor([-1.0, 2.0]))

    def test_invalid_child_lowering_state_fails_early(self) -> None:
        """A child graph without parent linkage must fail before forward execution."""
        from protofx.ops._registry import register_op

        @register_op("_EmitterChildGraphInvalid")
        def _handler(node, args, fx_graph, module):
            helper = getattr(module, "_protofx_child_graph_emitter", None)
            if helper is None:
                raise ValueError("missing internal child graph emitter helper")

            child_graph = node.subgraphs.get("then_branch")
            if not isinstance(child_graph, Graph):
                raise ValueError("missing child graph for _EmitterChildGraphInvalid")

            branch, _ = helper.make_callable(owner_node=node, branch_name="then_branch", child_graph=child_graph)
            packed = fx_graph.call_function(branch, args=(args[0],))
            return [fx_graph.call_function(operator.getitem, args=(packed, 0))]

        parent = Graph(name="parent")
        x = parent.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)), name="x")
        child = Graph(name="orphan")
        child_x = child.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)), name="x")
        child_identity = child.make_node(
            op_type="Identity",
            inputs=[child_x],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(2,))],
        )
        child.set_graph_outputs([child_identity.outputs[0]])

        node = parent.make_node(
            op_type="_EmitterChildGraphInvalid",
            inputs=[x],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(2,))],
            subgraphs={"then_branch": child},
        )
        parent.set_graph_outputs([node.outputs[0]])

        with pytest.raises(ValueError, match="missing parent linkage"):
            emit_graph(parent)


class TestAuthoritativeShapePreconditions:
    """Verify emit_graph handlers consume authoritative derived shapes."""

    def test_emit_graph_uses_authoritative_shape_for_flatten(self) -> None:
        graph = Graph(name="flatten_precondition")
        x = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3, 4)), name="x")
        node = graph.make_node(
            op_type="Flatten",
            inputs=[x],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(999, 999))],
            attributes={"axis": 1},
        )
        graph.set_graph_outputs([node.outputs[0]])
        set_derived_shape(node.outputs[0], (2, 12))

        gm = emit_graph(graph)
        (result,) = gm(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))
        assert result.shape == (2, 12)
