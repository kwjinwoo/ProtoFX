"""Internal ONNX-to-IR conversion logic.

This module implements the pipeline from ``onnx.ModelProto`` to ``ir.Graph``.
All ONNX-aware parsing, normalization, and deduplication happens here so the
emitter never touches raw protobuf structures.
"""

from __future__ import annotations

import onnx

from protofx.ir.dim import Dim
from protofx.ir.graph import Graph
from protofx.ir.node import AttributeValue
from protofx.ir.shape import Shape
from protofx.ir.tensor_type import TensorType
from protofx.ir.value import Value
from protofx.utils.dtype import onnx_dtype_to_ir

# ------------------------------------------------------------------
# Type parsing helpers
# ------------------------------------------------------------------


def _parse_dim(dim: onnx.TensorShapeProto.Dimension) -> Dim:
    """Convert a single ONNX dimension to an IR ``Dim``.

    Args:
        dim: An ONNX ``TensorShapeProto.Dimension``.

    Returns:
        An ``int`` for concrete values, a ``str`` for symbolic parameters,
        or ``None`` for entirely unknown dimensions.
    """
    if dim.HasField("dim_param"):
        return dim.dim_param
    if dim.HasField("dim_value"):
        return dim.dim_value
    return None


def _parse_tensor_type(type_proto: onnx.TypeProto) -> TensorType:
    """Convert an ONNX ``TypeProto`` to an IR ``TensorType``.

    Args:
        type_proto: The ONNX type descriptor.

    Returns:
        A ``TensorType`` with mapped dtype and shape.
    """
    tensor_type = type_proto.tensor_type
    dtype = onnx_dtype_to_ir(tensor_type.elem_type)

    shape: Shape
    if tensor_type.HasField("shape"):
        shape = tuple(_parse_dim(d) for d in tensor_type.shape.dim)
    else:
        shape = None

    return TensorType(dtype=dtype, shape=shape)


def _tensor_proto_to_tensor_type(tp: onnx.TensorProto) -> TensorType:
    """Derive an IR ``TensorType`` from an ONNX ``TensorProto``.

    Args:
        tp: An ONNX tensor (initializer or constant).

    Returns:
        A ``TensorType`` with dtype and shape extracted from the tensor.
    """
    dtype = onnx_dtype_to_ir(tp.data_type)
    shape: Shape = tuple(int(d) for d in tp.dims)
    return TensorType(dtype=dtype, shape=shape)


# ------------------------------------------------------------------
# Attribute normalization
# ------------------------------------------------------------------


def _normalize_attribute(attr: onnx.AttributeProto) -> AttributeValue:
    """Convert an ONNX ``AttributeProto`` to a Python-native ``AttributeValue``.

    String attributes are kept as raw ``bytes`` (ONNX stores strings as bytes
    in protobuf). Repeated string attributes produce ``list[bytes]``.

    Args:
        attr: The ONNX attribute to normalize.

    Returns:
        A Python-native value matching the ``AttributeValue`` type alias.

    Raises:
        NotImplementedError: For TENSOR, GRAPH, and other unsupported types.
    """
    match attr.type:
        case onnx.AttributeProto.INT:
            return int(attr.i)
        case onnx.AttributeProto.FLOAT:
            return float(attr.f)
        case onnx.AttributeProto.STRING:
            return bytes(attr.s)
        case onnx.AttributeProto.INTS:
            return [int(v) for v in attr.ints]
        case onnx.AttributeProto.FLOATS:
            return [float(v) for v in attr.floats]
        case onnx.AttributeProto.STRINGS:
            return [bytes(s) for s in attr.strings]
        case _:
            type_name = onnx.AttributeProto.AttributeType.Name(attr.type)
            msg = f"unsupported ONNX attribute type: {type_name}"
            raise NotImplementedError(msg)


# ------------------------------------------------------------------
# Import stages
# ------------------------------------------------------------------


def _import_initializers(
    graph: Graph,
    graph_proto: onnx.GraphProto,
) -> set[str]:
    """Import ONNX initializers into the IR graph.

    Args:
        graph: The IR graph being built.
        graph_proto: The source ONNX graph.

    Returns:
        Set of initializer names for input deduplication.
    """
    init_names: set[str] = set()
    for tp in graph_proto.initializer:
        tt = _tensor_proto_to_tensor_type(tp)
        data = onnx.numpy_helper.to_array(tp)
        graph.add_initializer(tensor_type=tt, data=data, name=tp.name)
        init_names.add(tp.name)
    return init_names


def _import_inputs(
    graph: Graph,
    graph_proto: onnx.GraphProto,
    init_names: set[str],
) -> None:
    """Import ONNX graph inputs into the IR graph, filtering initializer overlaps.

    In opset < 9 models, initializer names may also appear in ``graph.input``.
    Those duplicates are skipped so that IR invariant 7 (inputs and initializers
    remain distinct) is preserved.

    Args:
        graph: The IR graph being built.
        graph_proto: The source ONNX graph.
        init_names: Names already registered as initializers.
    """
    for vi in graph_proto.input:
        if vi.name in init_names:
            continue
        tt = _parse_tensor_type(vi.type)
        graph.add_input(tensor_type=tt, name=vi.name)


def _build_value_info_map(graph_proto: onnx.GraphProto) -> dict[str, onnx.TypeProto]:
    """Build a name-to-TypeProto map from value_info and graph outputs.

    Args:
        graph_proto: The source ONNX graph.

    Returns:
        A dict mapping value names to their ONNX ``TypeProto``.
    """
    vi_map: dict[str, onnx.TypeProto] = {}
    for vi in graph_proto.value_info:
        vi_map[vi.name] = vi.type
    for vi in graph_proto.output:
        vi_map[vi.name] = vi.type
    return vi_map


def _import_nodes(
    graph: Graph,
    graph_proto: onnx.GraphProto,
    value_registry: dict[str, Value],
    vi_map: dict[str, onnx.TypeProto],
) -> None:
    """Import ONNX nodes into the IR graph with use-def wiring.

    For each ONNX node, resolves input Values from the registry, creates
    output Values with type info from ``vi_map``, and wires use-def links
    via ``graph.make_node()``.

    Empty input names are mapped to ``SENTINEL`` values (omitted optionals).

    Args:
        graph: The IR graph being built.
        graph_proto: The source ONNX graph.
        value_registry: Mutable name-to-Value mapping for use-def resolution.
        vi_map: Name-to-TypeProto mapping for output type resolution.
    """
    for node_proto in graph_proto.node:
        # Resolve inputs
        inputs: list[Value] = []
        for input_name in node_proto.input:
            if not input_name:
                inputs.append(graph.add_sentinel())
            else:
                inputs.append(value_registry[input_name])

        # Resolve output types
        output_types: list[TensorType] = []
        output_names: list[str | None] = []
        for output_name in node_proto.output:
            if output_name in vi_map:
                output_types.append(_parse_tensor_type(vi_map[output_name]))
            else:
                output_types.append(TensorType(dtype=None, shape=None))
            output_names.append(output_name or None)

        # Normalize attributes
        attributes: dict[str, AttributeValue] = {}
        for attr in node_proto.attribute:
            attributes[attr.name] = _normalize_attribute(attr)

        # Create ir.Node
        ir_node = graph.make_node(
            op_type=node_proto.op_type,
            inputs=inputs,
            output_types=output_types,
            domain=node_proto.domain,
            attributes=attributes,
            name=node_proto.name or None,
            output_names=output_names,
        )

        # Register outputs in the value registry
        for out_value in ir_node.outputs:
            if out_value.name:
                value_registry[out_value.name] = out_value


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------


def import_model(model_proto: onnx.ModelProto) -> Graph:
    """Convert an ``onnx.ModelProto`` into an ``ir.Graph``.

    This is the main importer entry point. It parses graph inputs,
    initializers, and nodes from the ONNX model and produces a
    graph-valid IR representation.

    Args:
        model_proto: The source ONNX model.

    Returns:
        A populated ``ir.Graph``.
    """
    graph_proto = model_proto.graph
    graph = Graph(name=graph_proto.name or None)

    # Phase 1: initializers first (needed for input dedup)
    init_names = _import_initializers(graph, graph_proto)

    # Phase 2: graph inputs (filtered against initializer names)
    _import_inputs(graph, graph_proto, init_names)

    # Phase 3: build value registry from inputs + initializers
    value_registry: dict[str, Value] = {}
    for v in graph.inputs:
        if v.name:
            value_registry[v.name] = v
    for v in graph.initializers:
        if v.name:
            value_registry[v.name] = v

    # Phase 4: import nodes
    vi_map = _build_value_info_map(graph_proto)
    _import_nodes(graph, graph_proto, value_registry, vi_map)

    # Phase 5: set graph outputs
    graph_outputs: list[Value] = []
    for out_vi in graph_proto.output:
        if out_vi.name in value_registry:
            graph_outputs.append(value_registry[out_vi.name])
    graph.set_graph_outputs(graph_outputs)

    return graph
