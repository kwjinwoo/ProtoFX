"""Tests for IR-level symbolic shape propagation."""

from protofx.ir import DType, Graph, TensorType
from protofx.ir.derived_shape import get_authoritative_shape
from protofx.ir.shape_propagation import propagate_shapes


def test_propagate_identity_overrides_seed_metadata() -> None:
    """Identity output shape should be derived from its input shape."""
    graph = Graph(name="identity_graph")
    x = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)), name="x")
    node = graph.make_node(
        op_type="Identity",
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(999, 999))],
    )
    graph.set_graph_outputs([node.outputs[0]])

    propagate_shapes(graph)

    assert get_authoritative_shape(node.outputs[0]) == (2, 3)


def test_propagate_add_broadcast_shape() -> None:
    """Elementwise Add should derive broadcasted output shape."""
    graph = Graph(name="add_graph")
    lhs = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)), name="lhs")
    rhs = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1, 3)), name="rhs")
    node = graph.make_node(
        op_type="Add",
        inputs=[lhs, rhs],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(111, 222))],
    )
    graph.set_graph_outputs([node.outputs[0]])

    propagate_shapes(graph)

    assert get_authoritative_shape(node.outputs[0]) == (2, 3)
