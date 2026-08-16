import json
import pickle

from fastapi import APIRouter, Depends, Header, Request, Response

from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core.logger import get_sub_logger
from nekro_agent.core.os_env import OsEnv
from nekro_agent.schemas.errors import NotFoundError, UnauthorizedError
from nekro_agent.schemas.rpc import RPCRequest
from nekro_agent.services.message_service import message_service
from nekro_agent.services.plugin.collector import plugin_collector
from nekro_agent.services.plugin.schema import SandboxMethodType
from nekro_agent.services.plugin.utils import get_sandbox_method_type
from nekro_agent.services.rpc_service import decode_rpc_request, execute_rpc_method
from nekro_agent.services.sandbox import session_registry

logger = get_sub_logger("rpc_bridge")
router = APIRouter(prefix="/ext", tags=["Tools"])


async def verify_rpc_token(x_rpc_token: str = Header(...)):
    """验证 RPC 调用令牌"""
    if not OsEnv.RPC_SECRET_KEY or x_rpc_token != OsEnv.RPC_SECRET_KEY:
        logger.warning("非法的 RPC 调用令牌")
        raise UnauthorizedError
    return True


@router.post("/rpc_exec", summary="RPC 命令执行", dependencies=[Depends(verify_rpc_token)])
async def rpc_exec(
    container_key: str,
    from_chat_key: str,
    data: Request,
    x_container_token: str = Header(default=""),
) -> Response:
    # RPC_SECRET_KEY 是全局共享的，只能证明"调用来自某个沙盒"；而沙盒里跑的是模型生成的
    # 代码，它可以任意填写查询串。身份必须来自服务端登记的一次性容器令牌，
    # 查询串里的 container_key / from_chat_key 一律只作日志用途。
    session = session_registry.resolve(x_container_token)
    if session is None:
        logger.warning(f"RPC 调用未携带有效容器令牌，已拒绝 (自称 chat_key={from_chat_key!r})")
        raise UnauthorizedError
    if from_chat_key != session.chat_key or container_key != session.container_key:
        logger.warning(
            f"RPC 调用自报身份与登记不符，按登记身份执行: "
            f"自称 ({container_key!r}, {from_chat_key!r}) 实为 ({session.container_key!r}, {session.chat_key!r})",
        )

    rpc_request: RPCRequest = decode_rpc_request(await data.body())

    logger.info(f"收到 RPC 执行请求: {rpc_request.method}")

    method = plugin_collector.get_method(rpc_request.method)
    if not method:
        raise NotFoundError(resource="RPC 方法")
    method_type: SandboxMethodType = get_sandbox_method_type(method=method)

    ctx: AgentCtx = await AgentCtx.create_by_chat_key(
        chat_key=session.chat_key,
        container_key=session.container_key,
    )
    args = [ctx, *rpc_request.args] if rpc_request.args else [ctx]
    kwargs = rpc_request.kwargs or {}

    result, error_message = await execute_rpc_method(method, args, kwargs)

    if method_type in [SandboxMethodType.AGENT, SandboxMethodType.BEHAVIOR]:
        await message_service.push_system_message(chat_key=session.chat_key, agent_messages=str(result))
    if method_type == SandboxMethodType.MULTIMODAL_AGENT:
        result = f"<AGENT_RESULT>{json.dumps(result, ensure_ascii=False)}</AGENT_RESULT>"
    return Response(
        content=error_message or pickle.dumps(result),
        media_type="application/octet-stream",
        headers={"Method-Type": method_type.value, "Run-Error": "True" if error_message else "False"},
    )
