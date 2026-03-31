"""ProtoFX ONNX op handler registry."""

# Import handler modules so @register_op decorators execute at package load time.
import protofx.ops.elementwise as elementwise  # noqa: F401
import protofx.ops.reduction as reduction  # noqa: F401
import protofx.ops.tensor as tensor  # noqa: F401
from protofx.ops._registry import dispatch_op, register_op

__all__ = ["dispatch_op", "register_op"]
