"""ProtoFX ONNX op handler registry."""

# Import handler modules so @register_op decorators execute at package load time.
import protofx.ops.activation as activation  # noqa: F401
import protofx.ops.conv as conv  # noqa: F401
import protofx.ops.elementwise as elementwise  # noqa: F401
import protofx.ops.linalg as linalg  # noqa: F401
import protofx.ops.normalization as normalization  # noqa: F401
import protofx.ops.pooling as pooling  # noqa: F401
import protofx.ops.reduction as reduction  # noqa: F401
import protofx.ops.tensor as tensor  # noqa: F401
from protofx.ops._registry import dispatch_op, list_registry, register_op

__all__ = ["dispatch_op", "list_registry", "register_op"]
