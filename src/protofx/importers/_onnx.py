"""Internal ONNX-to-IR conversion logic.

This module implements the pipeline from ``onnx.ModelProto`` to ``ir.Graph``.
All ONNX-aware parsing, normalization, and deduplication happens here so the
emitter never touches raw protobuf structures.
"""

from __future__ import annotations

import math

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


def _inline_constant(
    graph: Graph,
    node_proto: onnx.NodeProto,
    value_registry: dict[str, Value],
) -> None:
    """Inline an ONNX Constant op as a ``CONSTANT`` value (no ir.Node created).

    Extracts the ``value`` TENSOR attribute, converts it to a numpy array,
    and creates a ``CONSTANT`` ir.Value registered in the value registry.

    Args:
        graph: The IR graph being built.
        node_proto: The ONNX Constant node proto.
        value_registry: Mutable name-to-Value mapping.
    """
    tensor_attr: onnx.TensorProto | None = None
    for attr in node_proto.attribute:
        if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
            tensor_attr = attr.t
            break

    if tensor_attr is None:
        msg = f"Constant node {node_proto.name!r} missing 'value' TENSOR attribute"
        raise ValueError(msg)

    data = onnx.numpy_helper.to_array(tensor_attr)
    tt = _tensor_proto_to_tensor_type(tensor_attr)
    output_name = node_proto.output[0] if node_proto.output else None

    const_value = graph.add_constant(tensor_type=tt, data=data, name=output_name)
    if output_name:
        value_registry[output_name] = const_value


def _normalize_conv_auto_pad(
    attributes: dict[str, AttributeValue],
    node_proto: onnx.NodeProto,
    value_registry: dict[str, Value],
    vi_map: dict[str, onnx.TypeProto],
) -> None:
    """Normalize ``auto_pad`` on Conv/ConvTranspose to explicit ``pads``.

    Converts ``VALID`` to all-zeros pads and ``SAME_UPPER``/``SAME_LOWER``
    to computed explicit padding values. After normalization ``auto_pad``
    is set to ``b"NOTSET"``.

    For ``SAME_*`` modes the input spatial dimensions must be statically
    known (concrete integers).

    Args:
        attributes: Mutable attribute dict to normalize in-place.
        node_proto: The source ONNX node proto.
        value_registry: Name-to-Value mapping for resolving input shapes.
        vi_map: Name-to-TypeProto mapping for resolving shapes.

    Raises:
        NotImplementedError: If ``SAME_*`` is requested but spatial dimensions
            are not statically known.
    """
    auto_pad_raw = attributes.get("auto_pad")
    if auto_pad_raw is None:
        return

    # ONNX stores string attrs as bytes
    auto_pad = auto_pad_raw.decode() if isinstance(auto_pad_raw, bytes) else str(auto_pad_raw)

    if auto_pad == "NOTSET":
        return

    # Determine spatial rank from weight shape
    w_name = node_proto.input[1] if len(node_proto.input) > 1 else ""
    kernel_shape: list[int] | None = None

    # Try kernel_shape attribute first
    ks_attr = attributes.get("kernel_shape")
    if ks_attr is not None:
        kernel_shape = [int(v) for v in ks_attr]  # type: ignore[union-attr]

    # Fallback: derive from weight tensor shape
    if kernel_shape is None and w_name:
        w_shape: tuple[int, ...] | None = None
        if w_name in value_registry and value_registry[w_name].tensor_type.shape is not None:
            w_shape = tuple(value_registry[w_name].tensor_type.shape)  # type: ignore[arg-type]
        elif w_name in vi_map:
            tp = vi_map[w_name].tensor_type
            if tp.HasField("shape"):
                w_shape = tuple(int(d.dim_value) for d in tp.shape.dim)
        if w_shape is not None:
            # Weight shape: Conv = (OC, IC/g, *kernel), ConvTranspose = (IC, OC/g, *kernel)
            kernel_shape = list(w_shape[2:])

    if kernel_shape is None:
        msg = f"{node_proto.op_type}: cannot determine kernel_shape for auto_pad normalization"
        raise NotImplementedError(msg)

    spatial_rank = len(kernel_shape)

    if auto_pad == "VALID":
        attributes["pads"] = [0] * (2 * spatial_rank)
        attributes["auto_pad"] = b"NOTSET"
        return

    # SAME_UPPER / SAME_LOWER need input spatial shape
    x_name = node_proto.input[0] if len(node_proto.input) > 0 else ""
    input_spatial: list[int] | None = None

    if x_name in value_registry and value_registry[x_name].tensor_type.shape is not None:
        full_shape = value_registry[x_name].tensor_type.shape
        input_spatial = [int(d) for d in full_shape[2:]]  # type: ignore[union-attr]
    elif x_name in vi_map:
        tp = vi_map[x_name].tensor_type
        if tp.HasField("shape"):
            dims = [d.dim_value for d in tp.shape.dim]
            input_spatial = [int(d) for d in dims[2:]]

    if input_spatial is None or len(input_spatial) != spatial_rank:
        msg = f"{node_proto.op_type}: SAME_* auto_pad requires static spatial dimensions"
        raise NotImplementedError(msg)

    strides = [int(v) for v in attributes.get("strides", [1] * spatial_rank)]  # type: ignore[union-attr]
    dilations = [int(v) for v in attributes.get("dilations", [1] * spatial_rank)]  # type: ignore[union-attr]

    pads_begin: list[int] = []
    pads_end: list[int] = []

    for i in range(spatial_rank):
        effective_kernel = kernel_shape[i] + (kernel_shape[i] - 1) * (dilations[i] - 1)
        out_size = math.ceil(input_spatial[i] / strides[i])
        total_pad = max(0, (out_size - 1) * strides[i] + effective_kernel - input_spatial[i])

        if auto_pad == "SAME_UPPER":
            pad_begin = total_pad // 2
            pad_end = total_pad - pad_begin
        else:  # SAME_LOWER
            pad_end = total_pad // 2
            pad_begin = total_pad - pad_end

        pads_begin.append(pad_begin)
        pads_end.append(pad_end)

    # ONNX pads format: [begin_d0, begin_d1, ..., end_d0, end_d1, ...]
    attributes["pads"] = pads_begin + pads_end
    attributes["auto_pad"] = b"NOTSET"


def _import_nodes(
    graph: Graph,
    graph_proto: onnx.GraphProto,
    value_registry: dict[str, Value],
    vi_map: dict[str, onnx.TypeProto],
    *,
    default_opset: int | None = None,
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
        default_opset: Default domain opset version from the model.
    """
    for node_proto in graph_proto.node:
        # Handle Constant op inlining
        if node_proto.op_type == "Constant" and node_proto.domain in ("", "ai.onnx"):
            _inline_constant(graph, node_proto, value_registry)
            continue

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

        # Normalize Conv/ConvTranspose auto_pad to explicit pads
        if node_proto.op_type in ("Conv", "ConvTranspose") and node_proto.domain in ("", "ai.onnx"):
            _normalize_conv_auto_pad(attributes, node_proto, value_registry, vi_map)

        # Resolve opset version: use node-level domain opset or model default
        opset_version = default_opset if node_proto.domain in ("", "ai.onnx") else None

        # Create ir.Node
        ir_node = graph.make_node(
            op_type=node_proto.op_type,
            inputs=inputs,
            output_types=output_types,
            domain=node_proto.domain,
            opset_version=opset_version,
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
    # Phase 0: run ONNX shape inference to fill missing value_info
    try:
        model_proto = onnx.shape_inference.infer_shapes(model_proto)
    except Exception:
        pass  # Inference failure is non-fatal; missing types become None

    graph_proto = model_proto.graph
    graph = Graph(name=graph_proto.name or None)

    # Extract default domain opset version
    default_opset: int | None = None
    for opset in model_proto.opset_import:
        if opset.domain in ("", "ai.onnx"):
            default_opset = opset.version
            break

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
    _import_nodes(graph, graph_proto, value_registry, vi_map, default_opset=default_opset)

    # Phase 5: set graph outputs
    graph_outputs: list[Value] = []
    for out_vi in graph_proto.output:
        if out_vi.name in value_registry:
            graph_outputs.append(value_registry[out_vi.name])
    graph.set_graph_outputs(graph_outputs)

    # Phase 6: validate IR invariants (fail-fast contract)
    graph.validate()

    return graph
