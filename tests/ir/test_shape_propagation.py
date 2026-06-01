"""Tests for IR-level symbolic shape propagation."""

import numpy as np

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


def test_propagate_flatten_overrides_seed_metadata() -> None:
    """Flatten output shape should be derived from authoritative input metadata."""
    graph = Graph(name="flatten_graph")
    x = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3, 4)), name="x")
    node = graph.make_node(
        op_type="Flatten",
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(99, 99))],
        attributes={"axis": 1},
    )
    graph.set_graph_outputs([node.outputs[0]])

    propagate_shapes(graph)

    assert get_authoritative_shape(node.outputs[0]) == (2, 12)


def test_propagate_reduce_sum_overrides_seed_metadata() -> None:
    """ReduceSum output shape should follow axes and keepdims semantics."""
    graph = Graph(name="reduce_graph")
    x = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3, 4)), name="x")
    node = graph.make_node(
        op_type="ReduceSum",
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(111, 222, 333))],
        attributes={"axes": [1], "keepdims": 0},
    )
    graph.set_graph_outputs([node.outputs[0]])

    propagate_shapes(graph)

    assert get_authoritative_shape(node.outputs[0]) == (2, 4)


def test_propagate_matmul_overrides_seed_metadata() -> None:
    """MatMul output shape should derive from input matrix shapes."""
    graph = Graph(name="matmul_graph")
    lhs = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)), name="lhs")
    rhs = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(3, 5)), name="rhs")
    node = graph.make_node(
        op_type="MatMul",
        inputs=[lhs, rhs],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(77, 88))],
    )
    graph.set_graph_outputs([node.outputs[0]])

    propagate_shapes(graph)

    assert get_authoritative_shape(node.outputs[0]) == (2, 5)


def test_propagate_conv_overrides_seed_metadata() -> None:
    """Conv output shape should derive from input shape, kernel, and attributes."""
    graph = Graph(name="conv_graph")
    x = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1, 3, 8, 8)), name="x")
    w = graph.add_initializer(
        tensor_type=TensorType(dtype=DType.FLOAT32, shape=(4, 3, 3, 3)),
        data=np.ones((4, 3, 3, 3), dtype=np.float32),
        name="w",
    )
    node = graph.make_node(
        op_type="Conv",
        inputs=[x, w],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(9, 9, 9, 9))],
        attributes={"strides": [2, 2], "pads": [1, 1, 1, 1], "dilations": [1, 1], "group": 1},
    )
    graph.set_graph_outputs([node.outputs[0]])

    propagate_shapes(graph)

    assert get_authoritative_shape(node.outputs[0]) == (1, 4, 4, 4)


def test_propagate_transpose_overrides_seed_metadata_with_symbolic_dim() -> None:
    """Transpose should permute symbolic shape metadata from authoritative inputs."""
    graph = Graph(name="transpose_graph")
    x = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=("n", 3, 5)), name="x")
    node = graph.make_node(
        op_type="Transpose",
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(99, 99, 99))],
        attributes={"perm": [1, 0, 2]},
    )
    graph.set_graph_outputs([node.outputs[0]])

    propagate_shapes(graph)

    assert get_authoritative_shape(node.outputs[0]) == (3, "n", 5)


def test_propagate_unsqueeze_overrides_seed_metadata_with_partial_unknown_shape() -> None:
    """Unsqueeze should insert singleton dims while preserving unknown input dims."""
    graph = Graph(name="unsqueeze_graph")
    x = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(None, 3)), name="x")
    node = graph.make_node(
        op_type="Unsqueeze",
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(99, 99, 99, 99))],
        attributes={"axes": [0, 2]},
    )
    graph.set_graph_outputs([node.outputs[0]])

    propagate_shapes(graph)

    assert get_authoritative_shape(node.outputs[0]) == (1, None, 1, 3)


def test_propagate_concat_overrides_seed_metadata_with_symbolic_mismatch() -> None:
    """Concat should derive axis sum and degrade non-axis symbolic mismatches to unknown."""
    graph = Graph(name="concat_graph")
    lhs = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=("lhs_batch", 2, 4)), name="lhs")
    rhs = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=("rhs_batch", 3, 4)), name="rhs")
    node = graph.make_node(
        op_type="Concat",
        inputs=[lhs, rhs],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(99, 99, 99))],
        attributes={"axis": 1},
    )
    graph.set_graph_outputs([node.outputs[0]])

    propagate_shapes(graph)

    assert get_authoritative_shape(node.outputs[0]) == (None, 5, 4)


def test_propagate_conv_transpose_overrides_seed_metadata() -> None:
    """ConvTranspose should derive output channels and spatial dimensions from inputs and attrs."""
    graph = Graph(name="conv_transpose_graph")
    x = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1, 4, 5, 5)), name="x")
    w = graph.add_initializer(
        tensor_type=TensorType(dtype=DType.FLOAT32, shape=(4, 3, 3, 3)),
        data=np.ones((4, 3, 3, 3), dtype=np.float32),
        name="w",
    )
    node = graph.make_node(
        op_type="ConvTranspose",
        inputs=[x, w],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(99, 99, 99, 99))],
        attributes={"strides": [2, 2], "pads": [1, 1, 1, 1], "dilations": [1, 1], "output_padding": [1, 1]},
    )
    graph.set_graph_outputs([node.outputs[0]])

    propagate_shapes(graph)

    assert get_authoritative_shape(node.outputs[0]) == (1, 3, 10, 10)


def test_propagate_maxpool_overrides_seed_metadata_with_symbolic_spatial_dim() -> None:
    """MaxPool should preserve unknown spatial dims when static derivation is impossible."""
    graph = Graph(name="maxpool_graph")
    x = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1, 3, "h", 8)), name="x")
    node = graph.make_node(
        op_type="MaxPool",
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(99, 99, 99, 99))],
        attributes={"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]},
    )
    graph.set_graph_outputs([node.outputs[0]])

    propagate_shapes(graph)

    assert get_authoritative_shape(node.outputs[0]) == (1, 3, None, 4)


def test_propagate_average_pool_overrides_seed_metadata() -> None:
    """AveragePool should derive static output shape from kernel and strides."""
    graph = Graph(name="average_pool_graph")
    x = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1, 3, 8, 8)), name="x")
    node = graph.make_node(
        op_type="AveragePool",
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(99, 99, 99, 99))],
        attributes={"kernel_shape": [2, 2], "strides": [2, 2]},
    )
    graph.set_graph_outputs([node.outputs[0]])

    propagate_shapes(graph)

    assert get_authoritative_shape(node.outputs[0]) == (1, 3, 4, 4)


def test_propagate_global_average_pool_overrides_seed_metadata_with_partial_unknown() -> None:
    """GlobalAveragePool should collapse spatial dims to singleton dimensions."""
    graph = Graph(name="global_average_pool_graph")
    x = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=("n", 16, None, 7)), name="x")
    node = graph.make_node(
        op_type="GlobalAveragePool",
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(99, 99, 99, 99))],
    )
    graph.set_graph_outputs([node.outputs[0]])

    propagate_shapes(graph)

    assert get_authoritative_shape(node.outputs[0]) == ("n", 16, 1, 1)


def test_propagate_squeeze_explicit_axis_with_symbolic_dim_stays_unknown() -> None:
    """Squeeze must not drop explicit symbolic axis without proof of ``== 1``."""
    graph = Graph(name="squeeze_symbolic_unknown_graph")
    x = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=("n", 3, 1)), name="x")
    node = graph.make_node(
        op_type="Squeeze",
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(3, 1))],
        attributes={"axes": [0]},
    )
    graph.set_graph_outputs([node.outputs[0]])

    propagate_shapes(graph)

    assert get_authoritative_shape(node.outputs[0]) is None


def test_propagate_squeeze_explicit_axis_with_unknown_dim_stays_unknown() -> None:
    """Squeeze must not drop explicit unknown axis without proof of ``== 1``."""
    graph = Graph(name="squeeze_none_unknown_graph")
    x = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(None, 3, 1)), name="x")
    node = graph.make_node(
        op_type="Squeeze",
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(3, 1))],
        attributes={"axes": [0]},
    )
    graph.set_graph_outputs([node.outputs[0]])

    propagate_shapes(graph)

    assert get_authoritative_shape(node.outputs[0]) is None


def test_propagate_reshape_minus_one_without_divisibility_stays_unknown() -> None:
    """Reshape ``-1`` axis must remain unknown when divisibility is unproven."""
    graph = Graph(name="reshape_minus_one_non_divisible_graph")
    x = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)), name="x")
    shape = graph.add_initializer(
        tensor_type=TensorType(dtype=DType.INT64, shape=(2,)),
        data=np.array([-1, 4], dtype=np.int64),
        name="shape",
    )
    node = graph.make_node(
        op_type="Reshape",
        inputs=[x, shape],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(99, 4))],
    )
    graph.set_graph_outputs([node.outputs[0]])

    propagate_shapes(graph)

    assert get_authoritative_shape(node.outputs[0]) == (None, 4)
