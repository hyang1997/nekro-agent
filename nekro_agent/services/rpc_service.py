import asyncio
import io
import pickle
from typing import Any, NoReturn, Tuple

from pydantic import ValidationError as PydanticValidationError

from nekro_agent.core.logger import get_sub_logger
from nekro_agent.schemas.errors import ValidationError
from nekro_agent.schemas.rpc import RPCRequest

logger = get_sub_logger("rpc_bridge")


class RestrictedUnpickler(pickle.Unpickler):
    """只接受纯数据的 Unpickler。

    RPC 请求体来自沙盒容器，而沙盒里跑的是模型生成的代码——也就是 Discord 上任何人
    都能通过提示注入间接影响的代码。裸 pickle.loads 会在反序列化过程中执行
    __reduce__，且发生在 RPCRequest.model_validate 之前，所以"先解包再校验"这个顺序
    本身就把校验架空了；而 nekro_agent 容器挂着可读写的 docker.sock，一旦在这里执行
    代码就等于拿到宿主 WSL 的 root。

    所有代码执行路径（REDUCE / GLOBAL / STACK_GLOBAL）都必须先经 find_class 解析出
    一个可调用对象，这里一律拒绝即可全部挡掉；而 dict / list / tuple / str / int /
    float / bool / None 这些纯数据 opcode 根本不走 find_class，因此线路格式保持兼容，
    沙盒侧不需要任何改动。
    """

    def find_class(self, module: str, name: str) -> NoReturn:
        logger.error(f"RPC 请求尝试反序列化对象，已拒绝: {module}.{name}")
        raise pickle.UnpicklingError(f"unsupported object in RPC payload: {module}.{name}")


def safe_pickle_loads(raw_body: bytes) -> Any:
    """按纯数据反序列化 RPC 请求体"""
    return RestrictedUnpickler(io.BytesIO(raw_body)).load()


def decode_rpc_request(raw_body: bytes) -> RPCRequest:
    try:
        payload = safe_pickle_loads(raw_body)
    except (pickle.UnpicklingError, EOFError, AttributeError, ValueError, ImportError, IndexError) as e:
        raise ValidationError(reason="RPC 请求格式错误") from e
    try:
        return RPCRequest.model_validate(payload)
    except PydanticValidationError as e:
        raise ValidationError(reason=str(e)) from e


async def execute_rpc_method(method: Any, args: list[Any], kwargs: dict[str, Any]) -> Tuple[Any, str]:
    try:
        if asyncio.iscoroutinefunction(method):
            result = await method(*args, **kwargs)
        else:
            result = method(*args, **kwargs)
        return result, ""
    except Exception as e:
        return None, str(e)
