"""Internal ONNX-to-IR conversion logic.

This module implements the pipeline from ``onnx.ModelProto`` to ``ir.Graph``.
All ONNX-aware parsing, normalization, and deduplication happens here so the
emitter never touches raw protobuf structures.
"""

from __future__ import annotations

import math

import numpy as np
import onnx

from protofx.ir.dim import Dim
from protofx.ir.dtype import DType
from protofx.ir.graph import Graph
from protofx.ir.node import AttributeValue
from protofx.ir.shape import Shape
from protofx.ir.tensor_type import TensorType
from protofx.ir.value import Value, ValueKind
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


def _normalize_legacy_axes_input(
    graph: Graph,
    node_proto: onnx.NodeProto,
    inputs: list[Value],
    attributes: dict[str, AttributeValue],
    *,
    default_opset: int | None,
) -> None:
    """Normalize legacy Squeeze/Unsqueeze ``axes`` attributes to constant inputs.

    In default-domain opset 11-12, Squeeze and Unsqueeze encode axes as an
    attribute instead of an input tensor. ProtoFX canonicalizes those schema
    differences at import time so downstream IR consumers only observe the
    input-tensor form.

    Args:
        graph: The IR graph being built.
        node_proto: The source ONNX node proto.
        inputs: Mutable imported input list for the node.
        attributes: Mutable normalized attribute dict for the node.
        default_opset: Default-domain opset version from the model.
    """
    if default_opset not in (11, 12):
        return
    if node_proto.domain not in ("", "ai.onnx"):
        return
    if node_proto.op_type not in ("Squeeze", "Unsqueeze"):
        return

    axes_attr = attributes.pop("axes", None)
    if axes_attr is None:
        return

    axes_data = np.asarray([int(axis) for axis in axes_attr], dtype=np.int64)
    axes_value = graph.add_constant(
        tensor_type=TensorType(dtype=DType.INT64, shape=(len(axes_data),)),
        data=axes_data,
    )
    inputs.append(axes_value)


def _reorder_child_capture_inputs(
    child_graph: Graph,
    *,
    child_capture_names: list[str],
    normalized_capture_order: list[str],
    owner_label: str,
) -> None:
    """Apply normalized capture input ordering to a child graph.

    Args:
        child_graph: Imported child graph to reorder.
        child_capture_names: Capture names materialized in this child graph.
        normalized_capture_order: Parent-scoped normalized capture ordering.
        owner_label: Diagnostic owner label for fail-fast errors.
    """
    if not child_capture_names:
        return

    capture_name_set = set(child_capture_names)
    capture_values_by_name = {
        value.name: value for value in child_graph.inputs if value.name in capture_name_set and value.name is not None
    }
    if set(capture_values_by_name) != capture_name_set:
        missing = sorted(capture_name_set - set(capture_values_by_name))
        msg = f"{owner_label}: unresolved capture inputs in child graph: {missing!r}"
        raise ValueError(msg)

    non_capture_inputs = [value for value in child_graph.inputs if value.name not in capture_name_set]
    ordered_capture_inputs = [
        capture_values_by_name[name] for name in normalized_capture_order if name in capture_name_set
    ]
    child_graph.inputs = non_capture_inputs + ordered_capture_inputs


def _normalize_if_capture_order(
    node_proto: onnx.NodeProto,
    value_registry: dict[str, Value],
    branch_captures: dict[str, tuple[Graph, list[str]]],
) -> list[str]:
    """Normalize If branch captures to a stable shared order.

    Args:
        node_proto: Source If node.
        value_registry: Parent graph name-to-value registry.
        branch_captures: Per-branch imported graph and capture names.

    Returns:
        Shared capture ordering to append to If node inputs.
    """
    then_entry = branch_captures.get("then_branch")
    else_entry = branch_captures.get("else_branch")
    if then_entry is None or else_entry is None:
        return []

    then_graph, then_captures = then_entry
    else_graph, else_captures = else_entry
    then_capture_set = set(then_captures)
    else_capture_set = set(else_captures)
    if then_capture_set != else_capture_set:
        msg = f"If node {node_proto.name or node_proto.op_type!r}: inconsistent branch captures"
        raise ValueError(msg)

    normalized_capture_order = [name for name in value_registry if name in then_capture_set]
    unresolved_capture_names = sorted(then_capture_set - set(normalized_capture_order))
    if unresolved_capture_names:
        unresolved_name = unresolved_capture_names[0]
        msg = f"If node {node_proto.name or node_proto.op_type!r}: unresolved capture {unresolved_name!r}"
        raise ValueError(msg)

    _reorder_child_capture_inputs(
        then_graph,
        child_capture_names=then_captures,
        normalized_capture_order=normalized_capture_order,
        owner_label=f"If node {node_proto.name or node_proto.op_type!r}",
    )
    _reorder_child_capture_inputs(
        else_graph,
        child_capture_names=else_captures,
        normalized_capture_order=normalized_capture_order,
        owner_label=f"If node {node_proto.name or node_proto.op_type!r}",
    )
    return normalized_capture_order


def _normalize_generic_child_capture_order(
    node_proto: onnx.NodeProto,
    value_registry: dict[str, Value],
    child_capture_entries: list[tuple[Graph, list[str]]],
) -> list[str]:
    """Normalize each generic child-graph capture order against parent registry.

    Args:
        node_proto: Source node carrying GRAPH/GRAPHS attributes.
        value_registry: Parent graph name-to-value registry.
        child_capture_entries: Child graphs paired with their materialized captures.

    Returns:
        Always returns an empty list because generic capture materialization is child-local only.
    """
    owner_label = f"node {node_proto.name or node_proto.op_type!r}"
    for child_graph, child_capture_names in child_capture_entries:
        capture_name_set = set(child_capture_names)
        normalized_capture_order = [name for name in value_registry if name in capture_name_set]
        unresolved_capture_names = sorted(capture_name_set - set(normalized_capture_order))
        if unresolved_capture_names:
            unresolved_name = unresolved_capture_names[0]
            msg = f"{owner_label}: unresolved capture {unresolved_name!r}"
            raise ValueError(msg)
        _reorder_child_capture_inputs(
            child_graph,
            child_capture_names=child_capture_names,
            normalized_capture_order=normalized_capture_order,
            owner_label=owner_label,
        )
    return []


def _shapes_provably_mismatch(
    lhs: tuple[int | str | None, ...] | None,
    rhs: tuple[int | str | None, ...] | None,
) -> bool:
    """Return whether two shapes have a provable mismatch.

    Args:
        lhs: First shape metadata.
        rhs: Second shape metadata.

    Returns:
        ``True`` when available metadata is sufficient to prove a mismatch.
    """
    if lhs is None or rhs is None:
        return False
    if len(lhs) != len(rhs):
        return True
    for left_dim, right_dim in zip(lhs, rhs, strict=True):
        if left_dim is None or right_dim is None:
            continue
        if isinstance(left_dim, str) or isinstance(right_dim, str):
            continue
        if left_dim != right_dim:
            return True
    return False


def _matches_tensor_type(candidate: Value, reference: Value) -> bool:
    """Return whether two values are shape/dtype compatible based on known metadata.

    Args:
        candidate: Candidate value metadata.
        reference: Reference value metadata.

    Returns:
        ``True`` when known metadata does not prove a mismatch.
    """
    candidate_dtype = candidate.tensor_type.dtype
    reference_dtype = reference.tensor_type.dtype
    if candidate_dtype is not None and reference_dtype is not None and candidate_dtype != reference_dtype:
        return False
    return not _shapes_provably_mismatch(candidate.tensor_type.shape, reference.tensor_type.shape)


def _normalize_loop_capture_order(
    node_proto: onnx.NodeProto,
    value_registry: dict[str, Value],
    node_inputs: list[Value],
    loop_entry: tuple[Graph, list[str]] | None,
) -> list[str]:
    """Normalize Loop body interfaces and captures to the MVP contract.

    Args:
        node_proto: Source Loop node.
        value_registry: Parent graph name-to-value registry.
        node_inputs: Parent Loop inputs resolved from ONNX positional operands.
        loop_entry: Imported Loop body graph and capture names.

    Returns:
        Loop capture ordering to append to parent Loop node inputs.
    """
    if loop_entry is None:
        return []

    body_graph, body_capture_names = loop_entry
    capture_name_set = set(body_capture_names)
    non_capture_inputs = [value for value in body_graph.inputs if value.name not in capture_name_set]

    carried_count = len(node_proto.output)
    expected_non_capture = 2 + carried_count
    owner_label = f"Loop node {node_proto.name or node_proto.op_type!r}"
    if len(node_inputs) < 2:
        msg = f"{owner_label}: expected explicit M and cond inputs"
        raise ValueError(msg)
    if node_inputs[1].kind == ValueKind.SENTINEL:
        msg = f"{owner_label}: omitted cond input is unsupported in MVP"
        raise ValueError(msg)
    if len(non_capture_inputs) != expected_non_capture:
        msg = (
            f"{owner_label}: body interface mismatch "
            f"(body_inputs={len(non_capture_inputs)}, expected={expected_non_capture})"
        )
        raise ValueError(msg)

    iteration_value = next(
        (
            value
            for value in non_capture_inputs
            if value.tensor_type.dtype == DType.INT64 and value.tensor_type.shape == ()
        ),
        None,
    )
    if iteration_value is None:
        msg = f"{owner_label}: body interface mismatch (missing iteration counter input)"
        raise ValueError(msg)

    remaining = [value for value in non_capture_inputs if value is not iteration_value]
    cond_reference = node_inputs[1] if len(node_inputs) > 1 else None
    if cond_reference is not None and cond_reference.tensor_type.dtype is not None:
        cond_value = next((value for value in remaining if _matches_tensor_type(value, cond_reference)), None)
    else:
        cond_value = next((value for value in remaining if value.tensor_type.dtype == DType.BOOL), None)
    if cond_value is None:
        msg = f"{owner_label}: body interface mismatch (missing incoming condition input)"
        raise ValueError(msg)

    remaining = [value for value in remaining if value is not cond_value]
    carried_inputs: list[Value] = []
    for carried_slot in range(carried_count):
        parent_slot = carried_slot + 2
        parent_value = node_inputs[parent_slot]
        matched = next((value for value in remaining if _matches_tensor_type(value, parent_value)), None)
        if matched is None:
            msg = f"{owner_label}: body interface mismatch at carried input {carried_slot}"
            raise ValueError(msg)
        carried_inputs.append(matched)
        remaining = [value for value in remaining if value is not matched]

    if remaining:
        msg = f"{owner_label}: body interface mismatch (unexpected body inputs)"
        raise ValueError(msg)

    normalized_capture_order = [name for name in value_registry if name in capture_name_set]
    unresolved_capture_names = sorted(capture_name_set - set(normalized_capture_order))
    if unresolved_capture_names:
        unresolved_name = unresolved_capture_names[0]
        msg = f"{owner_label}: unresolved capture {unresolved_name!r}"
        raise ValueError(msg)

    _reorder_child_capture_inputs(
        body_graph,
        child_capture_names=body_capture_names,
        normalized_capture_order=normalized_capture_order,
        owner_label=owner_label,
    )
    capture_values_by_name = {value.name: value for value in body_graph.inputs if value.name is not None}
    ordered_capture_inputs = [capture_values_by_name[name] for name in normalized_capture_order]
    body_graph.inputs = [iteration_value, cond_value, *carried_inputs, *ordered_capture_inputs]
    return normalized_capture_order


def _import_nodes(
    graph: Graph,
    graph_proto: onnx.GraphProto,
    value_registry: dict[str, Value],
    vi_map: dict[str, onnx.TypeProto],
    *,
    default_opset: int | None = None,
    parent_value_registry: dict[str, Value] | None = None,
    capture_order: list[str] | None = None,
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
            elif input_name in value_registry:
                inputs.append(value_registry[input_name])
            elif parent_value_registry is not None and input_name in parent_value_registry:
                captured = graph.add_input(
                    tensor_type=parent_value_registry[input_name].tensor_type,
                    name=input_name,
                )
                value_registry[input_name] = captured
                if capture_order is not None:
                    capture_order.append(input_name)
                inputs.append(captured)
            else:
                msg = f"unresolved input {input_name!r} for node {node_proto.op_type!r}"
                raise ValueError(msg)

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
        subgraphs: dict[str, Graph | tuple[Graph, ...]] = {}
        if_branch_captures: dict[str, tuple[Graph, list[str]]] = {}
        loop_body_capture: tuple[Graph, list[str]] | None = None
        generic_child_captures: list[tuple[Graph, list[str]]] = []
        for attr in node_proto.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                child_graph, child_captures = _import_graph_proto(
                    attr.g,
                    default_opset=default_opset,
                    parent=graph,
                    parent_value_registry=value_registry,
                )
                subgraphs[attr.name] = child_graph
                if node_proto.op_type == "If" and attr.name in {"then_branch", "else_branch"}:
                    if_branch_captures[attr.name] = (child_graph, child_captures)
                elif node_proto.op_type == "Loop" and attr.name == "body":
                    loop_body_capture = (child_graph, child_captures)
                else:
                    generic_child_captures.append((child_graph, child_captures))
                continue
            if attr.type == onnx.AttributeProto.GRAPHS:
                child_graphs: list[Graph] = []
                attr_capture_entries: list[tuple[Graph, list[str]]] = []
                for child_proto in attr.graphs:
                    child_graph, child_captures = _import_graph_proto(
                        child_proto,
                        default_opset=default_opset,
                        parent=graph,
                        parent_value_registry=value_registry,
                    )
                    child_graphs.append(child_graph)
                    attr_capture_entries.append((child_graph, child_captures))
                subgraphs[attr.name] = tuple(child_graphs)
                generic_child_captures.extend(attr_capture_entries)
                continue
            attributes[attr.name] = _normalize_attribute(attr)

        if node_proto.op_type == "If":
            normalized_capture_order = _normalize_if_capture_order(node_proto, value_registry, if_branch_captures)
            for capture_name in normalized_capture_order:
                inputs.append(value_registry[capture_name])
        elif node_proto.op_type == "Loop":
            normalized_capture_order = _normalize_loop_capture_order(
                node_proto, value_registry, inputs, loop_body_capture
            )
            for capture_name in normalized_capture_order:
                inputs.append(value_registry[capture_name])
        else:
            _normalize_generic_child_capture_order(
                node_proto,
                value_registry,
                generic_child_captures,
            )

        # Normalize Conv/ConvTranspose auto_pad to explicit pads
        if node_proto.op_type in ("Conv", "ConvTranspose") and node_proto.domain in ("", "ai.onnx"):
            _normalize_conv_auto_pad(attributes, node_proto, value_registry, vi_map)

        _normalize_legacy_axes_input(
            graph,
            node_proto,
            inputs,
            attributes,
            default_opset=default_opset,
        )

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
            subgraphs=subgraphs,
            name=node_proto.name or None,
            output_names=output_names,
        )

        # Register outputs in the value registry
        for out_value in ir_node.outputs:
            if out_value.name:
                value_registry[out_value.name] = out_value


def _import_graph_proto(
    graph_proto: onnx.GraphProto,
    *,
    default_opset: int | None,
    parent: Graph | None,
    parent_value_registry: dict[str, Value] | None,
) -> tuple[Graph, list[str]]:
    """Import one ONNX ``GraphProto`` into an IR graph.

    Args:
        graph_proto: The source ONNX graph.
        default_opset: Default domain opset version from the model.
        parent: Optional parent IR graph.
        parent_value_registry: Optional outer-scope values for capture normalization.

    Returns:
        A tuple ``(graph, capture_order)`` where ``capture_order`` lists outer-scope
        capture names materialized as explicit child-graph inputs.
    """
    graph = Graph(name=graph_proto.name or None, parent=parent)

    init_names = _import_initializers(graph, graph_proto)
    _import_inputs(graph, graph_proto, init_names)

    value_registry: dict[str, Value] = {}
    for value in graph.inputs:
        if value.name:
            value_registry[value.name] = value
    for value in graph.initializers:
        if value.name:
            value_registry[value.name] = value

    capture_order: list[str] = []
    vi_map = _build_value_info_map(graph_proto)
    _import_nodes(
        graph,
        graph_proto,
        value_registry,
        vi_map,
        default_opset=default_opset,
        parent_value_registry=parent_value_registry,
        capture_order=capture_order,
    )

    graph_outputs: list[Value] = []
    for out_vi in graph_proto.output:
        if out_vi.name in value_registry:
            graph_outputs.append(value_registry[out_vi.name])
    graph.set_graph_outputs(graph_outputs)
    return graph, capture_order


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

    # Extract default domain opset version
    default_opset: int | None = None
    for opset in model_proto.opset_import:
        if opset.domain in ("", "ai.onnx"):
            default_opset = opset.version
            break

    graph, _ = _import_graph_proto(
        model_proto.graph,
        default_opset=default_opset,
        parent=None,
        parent_value_registry=None,
    )

    # Phase 6: validate IR invariants (fail-fast contract)
    graph.validate()

    return graph
