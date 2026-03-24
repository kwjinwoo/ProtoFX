"""ProtoFX FX emitter — converts normalized ``ir.Graph`` into ``torch.fx.GraphModule``."""

from protofx.emitters._fx import emit_graph

__all__ = ["emit_graph"]
