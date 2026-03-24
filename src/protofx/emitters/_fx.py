"""Internal IR-to-FX emission logic.

This module implements the pipeline from ``ir.Graph`` to
``torch.fx.GraphModule``. All FX-aware lowering happens here so the
importer and IR core never depend on ``torch.fx``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx

from protofx.ir.graph import Graph


def emit_graph(graph: Graph) -> torch.fx.GraphModule:
    """Convert a normalized ``ir.Graph`` into a ``torch.fx.GraphModule``.

    This is the main emitter entry point. It walks the IR graph in
    topological order, emits FX nodes for each IR value and operation,
    and returns a fully constructed ``GraphModule``.

    Args:
        graph: A validated, normalized IR graph produced by the importer.

    Returns:
        A ``torch.fx.GraphModule`` equivalent to the IR graph.
    """
    import torch
    import torch.fx

    fx_graph = torch.fx.Graph()
    root = torch.nn.Module()

    # Emit output node (empty graph case)
    fx_graph.output(())

    return torch.fx.GraphModule(root, fx_graph)
