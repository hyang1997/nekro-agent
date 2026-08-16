import asyncio
import contextlib
import os
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

import aiodocker
from aiodocker.docker import DockerContainer

from nekro_agent.core.config import config
from nekro_agent.core.logger import get_sub_logger
from nekro_agent.core.os_env import (
    SANDBOX_PACKAGE_DIR,
    SANDBOX_PIP_CACHE_DIR,
    SANDBOX_SHARED_HOST_DIR,
    USER_UPLOAD_DIR,
    OsEnv,
)
from nekro_agent.models.db_exec_code import DBExecCode, ExecStopType
from nekro_agent.schemas.agent_ctx import AgentCtx
from nekro_agent.schemas.chat_message import ChatMessage
from nekro_agent.schemas.sandbox import SandboxCodeExtData
from nekro_agent.services.agent.openai import OpenAIResponse
from nekro_agent.services.agent.resolver import ParsedCodeRunData
from nekro_agent.tools.common_util import limited_text_output

from . import session_registry
from .ext_caller import CODE_PREAMBLE, get_api_caller_code

# 主机共享目录

logger = get_sub_logger("sandbox_runtime")
HOST_SHARED_DIR = (
    Path(SANDBOX_SHARED_HOST_DIR) if SANDBOX_SHARED_HOST_DIR.startswith("/") else Path(SANDBOX_SHARED_HOST_DIR).resolve()
)
# 用户上传目录
USER_UPLOAD_DIR = Path(USER_UPLOAD_DIR) if USER_UPLOAD_DIR.startswith("/") else Path(USER_UPLOAD_DIR).resolve()
# 主机pip缓存目录
HOST_PIP_CACHE_DIR = (
    Path(SANDBOX_PIP_CACHE_DIR) if SANDBOX_PIP_CACHE_DIR.startswith("/") else Path(SANDBOX_PIP_CACHE_DIR).resolve()
)
# 主机包目录
HOST_PACKAGE_DIR = Path(SANDBOX_PACKAGE_DIR) if SANDBOX_PACKAGE_DIR.startswith("/") else Path(SANDBOX_PACKAGE_DIR).resolve()

IMAGE_NAME = config.SANDBOX_IMAGE_NAME  # Docker 镜像名称
CONTAINER_SHARE_DIR = "/app/shared"  # 容器内共享目录 (读写)
CONTAINER_UPLOAD_DIR = "/app/uploads"  # 容器上传目录 (只读)
CONTAINER_WORK_DIR = "/app"  # 容器工作目录
CONTAINER_PIP_CACHE_DIR = "/app/.pip_cache"  # 容器pip缓存目录
CONTAINER_PACKAGE_DIR = "/app/packages"  # 容器包缓存目录

CODE_FILENAME = "run_script.py.code"  # 要执行的代码文件名
RUN_CODE_FILENAME = "run_script.py"  # 要执行的代码文件名

API_CALLER_FILENAME = "api_caller.py.code"  # 外部 API 调用器文件名
RUN_API_CALLER_FILENAME = "api_caller.py"  # 外部 API 调用器文件名

# 代码运行结束标记
CODE_RUN_END_FLAGS = {
    ExecStopType.NORMAL: "[SANDBOX_RUN_ENDS_WITH_NORMAL]",  # 正常结束 (exit code 0)
    ExecStopType.ERROR: "[SANDBOX_RUN_ENDS_WITH_ERROR]",  # 错误停止 (exit code 非0)
    ExecStopType.TIMEOUT: "[SANDBOX_RUN_ENDS_WITH_TIMEOUT]",  # 超时停止
    ExecStopType.AGENT: "[SANDBOX_RUN_ENDS_WITH_AGENT]",  # 代理停止 (exit code 8)
    ExecStopType.MANUAL: "[SANDBOX_RUN_ENDS_WITH_MANUAL]",  # 手动停止 (exit code 9)
    ExecStopType.MULTIMODAL_AGENT: "[SANDBOX_RUN_ENDS_WITH_MULTIMODAL_AGENT]",  # 多模态代理停止 (exit code 11)
}

EXEC_SCRIPT = f"""
rm -f {CONTAINER_WORK_DIR}/{RUN_CODE_FILENAME} &&
cp {CONTAINER_SHARE_DIR}/{CODE_FILENAME} {CONTAINER_WORK_DIR}/{RUN_CODE_FILENAME} &&
cp {CONTAINER_SHARE_DIR}/{API_CALLER_FILENAME} {CONTAINER_WORK_DIR}/{RUN_API_CALLER_FILENAME} &&
export MPLCONFIGDIR=/app/tmp/matplotlib &&
python {RUN_CODE_FILENAME}
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "{CODE_RUN_END_FLAGS[ExecStopType.NORMAL]}"
elif [ $exit_code -eq 8 ]; then
    echo "{CODE_RUN_END_FLAGS[ExecStopType.AGENT]}"
elif [ $exit_code -eq 9 ]; then
    echo "{CODE_RUN_END_FLAGS[ExecStopType.MANUAL]}"
elif [ $exit_code -eq 11 ]; then
    echo "{CODE_RUN_END_FLAGS[ExecStopType.MULTIMODAL_AGENT]}"
else
    echo "{CODE_RUN_END_FLAGS[ExecStopType.ERROR]}"
fi
"""

SPILL_DIR_NAME = ".stdout"  # 共享目录下存放完整输出的子目录
SPILL_KEEP_FILES = 20  # 每个频道保留的溢出文件数


def _write_output_spill(host_shared_dir: Path, output_text: str) -> Optional[str]:
    """把完整输出写入共享目录，返回沙盒内可见的相对路径

    截断只保留首尾，中间部分对模型永久丢失，而沙盒 stdout 是模型唯一的观察渠道。
    共享目录在下一轮沙盒里会挂载到同一位置，所以把完整输出放在这里，模型可以自己读回来。
    落盘失败不能把一次成功的执行变成失败，因此这里吞掉异常返回 None，调用方降级为纯截断提示。
    """
    try:
        spill_dir = host_shared_dir / SPILL_DIR_NAME
        spill_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{time.strftime('%H%M%S')}_{os.urandom(3).hex()}.txt"
        (spill_dir / filename).write_text(output_text, encoding="utf-8")
        with contextlib.suppress(Exception):
            Path.chmod(spill_dir, 0o777)
        # 只保留最近若干个，避免共享目录被历史输出撑大
        stale_files = sorted(spill_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)[SPILL_KEEP_FILES:]
        for stale in stale_files:
            with contextlib.suppress(Exception):
                stale.unlink()
    except Exception as e:
        logger.warning(f"沙盒输出落盘失败，本轮被截断的内容将无法恢复: {e}")
        return None
    return f"./shared/{SPILL_DIR_NAME}/{filename}"


class OutputTruncation(NamedTuple):
    """截断结果。截断与退出类型正交，所以独立上报而不是塞进 stop_type"""

    text: str  # 回传给模型的文本
    truncated: bool
    total_chars: int  # 截断前的原始长度
    spill_path: str  # 完整输出的落盘路径，未落盘为空串


def _build_final_output(output_text: str, output_limit: int, host_shared_dir: Path) -> OutputTruncation:
    """按上限截断输出，并在截断提示里给出恢复手段"""
    if len(output_text) <= output_limit:
        return OutputTruncation(output_text, False, len(output_text), "")

    spill_path = _write_output_spill(host_shared_dir, output_text) if config.SANDBOX_OUTPUT_SPILL else None
    # 只说"被截断了"没用，模型需要的是"怎么把它拿回来"。恢复手段必须跟着截断提示一起
    # 到达模型：写在系统提示里的静态说明到不了这个决策点，而这条提示恰好出现在它必须行动的时刻。
    if spill_path:
        slice_start = output_limit // 2
        recovery = (
            f" Full output saved to {spill_path} — to see the omitted middle, read a slice of it and exit(9),"
            f" e.g. print(open('{spill_path}').read()[{slice_start}:{slice_start + output_limit}]); exit(9)"
        )
    else:
        recovery = " The omitted middle is NOT recoverable — re-run printing only the part you actually need"
    logger.warning(
        f"沙盒输出超出上限被截断: {len(output_text)} 字符 > 上限 {output_limit}"
        f"{f'，完整输出已落盘 {spill_path}' if spill_path else '，未落盘，中间部分不可恢复'}",
    )
    text = limited_text_output(
        output_text,
        limit=output_limit,
        placeholder=f"...(output truncated: {len(output_text) - output_limit} of {len(output_text)} characters hidden.{recovery})...",
    )
    return OutputTruncation(text, True, len(output_text), spill_path or "")


# 频道沙盒活跃时间记录表
chat_key_sandbox_map: Dict[str, float] = {}

# 频道沙盒容器记录表
chat_key_sandbox_container_map: Dict[str, DockerContainer] = {}

# 频道清理任务记录表
chat_key_sandbox_cleanup_task_map: Dict[str, asyncio.Task] = {}

# 沙盒并发限制
semaphore = asyncio.Semaphore(config.SANDBOX_MAX_CONCURRENT)


def _sanitize_docker_name_part(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", value)
    return sanitized or "unknown"


async def limited_run_code(
    code_run_data: ParsedCodeRunData,
    from_chat_key: str,
    output_limit: Optional[int] = None,
    llm_response: Optional[OpenAIResponse] = None,
    chat_message: Optional[ChatMessage] = None,
    ctx: Optional[AgentCtx] = None,
    llm_retry_errors: Optional[list[str]] = None,
) -> Tuple[str, str, int]:
    """限制并发运行代码

    Args:
        code_run_data: 代码执行数据
        from_chat_key: 频道键
        output_limit: 输出限制，None 时取 config.SANDBOX_OUTPUT_LIMIT
        llm_response: LLM 响应
        chat_message: 聊天消息
        ctx: Agent 上下文
        llm_retry_errors: LLM 重试过程中产生的错误信息列表

    Returns:
        Tuple[str, str, int]: 最终输出结果、原始输出结果和退出类型
    """

    async with semaphore:
        return await run_code_in_sandbox(
            code_run_data=code_run_data,
            from_chat_key=from_chat_key,
            output_limit=config.SANDBOX_OUTPUT_LIMIT if output_limit is None else output_limit,
            llm_response=llm_response,
            chat_message=chat_message,
            ctx=ctx,
            llm_retry_errors=llm_retry_errors,
        )


async def run_code_in_sandbox(
    code_run_data: ParsedCodeRunData,
    from_chat_key: str,
    output_limit: int,
    llm_response: Optional[OpenAIResponse] = None,
    chat_message: Optional[ChatMessage] = None,
    ctx: Optional[AgentCtx] = None,
    llm_retry_errors: Optional[list[str]] = None,
) -> Tuple[str, str, int]:
    """在沙盒容器中运行代码并获取输出"""

    # 记录开始时间
    start_time = time.time()

    generation_time_ms = llm_response.generation_time_ms if llm_response else 0

    # container_key = f'{time.strftime("%Y%m%d%H%M%S")}_{os.urandom(4).hex()}'
    container_key = f"sandbox_{_sanitize_docker_name_part(from_chat_key)}"
    container_name = f"nekro-agent-sandbox-{container_key}-{os.urandom(4).hex()}"

    host_shared_dir = Path(HOST_SHARED_DIR / container_key)
    host_shared_dir.mkdir(parents=True, exist_ok=True)

    HOST_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    HOST_PIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 本次运行的一次性身份令牌。此刻只是写进 api_caller.py，容器启动前才登记生效
    container_token = secrets.token_urlsafe(32)

    # 写入预置依赖代码
    api_caller_file_path = Path(host_shared_dir) / API_CALLER_FILENAME
    api_caller_file_path.write_text(
        await get_api_caller_code(
            container_key=container_key,
            from_chat_key=from_chat_key,
            container_token=container_token,
            ctx=ctx,
        ),
        encoding="utf-8",
    )

    # 写入要执行的代码
    code_file_path = Path(host_shared_dir) / CODE_FILENAME
    code_file_path.write_text(f"{CODE_PREAMBLE.strip()}\n\n{code_run_data.code_content}", encoding="utf-8")

    # 设置目录权限
    try:
        Path.chmod(host_shared_dir, 0o777)
        logger.debug(f"设置目录权限: {host_shared_dir} 777")
        Path.chmod(HOST_PIP_CACHE_DIR, 0o777)
        logger.debug(f"设置目录权限: {HOST_PIP_CACHE_DIR} 777")
        Path.chmod(HOST_PACKAGE_DIR, 0o777)
        logger.debug(f"设置目录权限: {HOST_PACKAGE_DIR} 777")
    except Exception as e:
        logger.error(f"设置目录权限失败: {e}")

    # 清理过期任务
    if from_chat_key in chat_key_sandbox_cleanup_task_map:
        try:
            chat_key_sandbox_cleanup_task_map[from_chat_key].cancel()
            logger.debug(f"清理过期任务: {from_chat_key}")
        except Exception as e:
            logger.error(f"清理过期任务失败: {e}")
        del chat_key_sandbox_cleanup_task_map[from_chat_key]

    # 清理过期沙盒
    if from_chat_key in chat_key_sandbox_container_map:
        try:
            await chat_key_sandbox_container_map[from_chat_key].delete()
            logger.debug(f"清理过期沙盒: {from_chat_key} | {container_name}")
        except Exception as e:
            if "404" in str(e):
                logger.debug(f"沙盒容器已不存在: {from_chat_key} | {container_name}")
            else:
                logger.warning(f"清理过期沙盒失败: {e}")
        del chat_key_sandbox_container_map[from_chat_key]

    # 启动容器
    # 使用 try/finally 确保 Docker 客户端（及其底层 aiohttp UnixConnector）在使用后被正确关闭，
    # 防止连接泄漏导致连接池耗尽后 docker.containers.run() 永久挂起
    docker = aiodocker.Docker()
    # 令牌只在容器运行期间有效：登记在启动前，注销在退出后，不可复用
    session_registry.register(container_token, container_key, from_chat_key)
    try:
        container: DockerContainer = await docker.containers.run(
            name=container_name,
            config={
                "Image": IMAGE_NAME,
                "Cmd": ["bash", "-c", EXEC_SCRIPT],
                "HostConfig": {
                    "Binds": [
                        f"{HOST_PIP_CACHE_DIR}:{CONTAINER_PIP_CACHE_DIR}:rw",
                        f"{HOST_PACKAGE_DIR}:{CONTAINER_PACKAGE_DIR}:rw",
                        f"{host_shared_dir}:{CONTAINER_SHARE_DIR}:rw",
                        f"{USER_UPLOAD_DIR}/{_sanitize_docker_name_part(from_chat_key)}:{CONTAINER_UPLOAD_DIR}:ro",
                    ],
                    "Memory": 512 * 1024 * 1024,  # 内存限制 (512MB)
                    "NanoCPUs": 1000000000,  # CPU 限制 (1 core)
                    "SecurityOpt": (
                        []
                        if OsEnv.RUN_IN_DOCKER
                        else [
                            # "no-new-privileges",  # 禁止提升权限
                            "apparmor=unconfined",  # 禁止 AppArmor 配置
                        ]
                    ),
                    "NetworkMode": "bridge",
                    "ExtraHosts": ["host.docker.internal:host-gateway"],
                },
                "User": "nobody",  # 非特权用户
                "AutoRemove": True,
            },
        )
        chat_key_sandbox_container_map[from_chat_key] = container
        logger.debug(f"启动容器: {container_name} | ID: {container.id}")

        # 获取输出和退出类型
        output_text, stop_type = await run_container_with_timeout(
            container,
            config.SANDBOX_RUNNING_TIMEOUT,
        )
    finally:
        session_registry.unregister(container_token)
        await docker.close()

    # 记录执行耗时
    exec_time = int((time.time() - start_time) * 1000)  # 转换为毫秒
    # 记录总耗时（生成耗时 + 执行耗时）
    total_time = generation_time_ms + exec_time

    logger.debug(f"容器 {container_name} 输出: {limited_text_output(output_text)} | 退出类型: {stop_type}")

    # 沙盒共享目录超过 30 分钟未活动，则自动清理
    async def cleanup_container_shared_dir(box_last_active_time):
        nonlocal from_chat_key, container
        await asyncio.sleep(30 * 60)
        if box_last_active_time == chat_key_sandbox_map.get(from_chat_key):
            try:
                shutil.rmtree(host_shared_dir)
            except Exception as e:
                logger.error(f"清理容器共享目录时发生错误: {e}")
            with contextlib.suppress(Exception):
                await container.delete()  # 清理沙盒

    box_last_active_time = time.time()
    chat_key_sandbox_map[from_chat_key] = box_last_active_time
    chat_key_sandbox_cleanup_task_map[from_chat_key] = asyncio.create_task(
        cleanup_container_shared_dir(box_last_active_time),
    )

    truncation = _build_final_output(output_text, output_limit, host_shared_dir)
    final_output = truncation.text

    ext_data = ""
    if llm_response:
        # 截断是独立于退出类型的事实：exit 0 的成功执行也可能只让模型看到了 2% 的输出。
        # 单靠 outputs 字段事后分不出"输出本来就短"和"被砍掉了 39000 字"。
        ext_data_obj = SandboxCodeExtData.create_from_llm_response(llm_response, llm_retry_errors=llm_retry_errors)
        ext_data_obj.output_truncated = truncation.truncated
        ext_data_obj.output_chars_total = truncation.total_chars
        ext_data_obj.output_spill_path = truncation.spill_path
        ext_data = ext_data_obj.model_dump_json()

    await DBExecCode.create(
        chat_key=from_chat_key,
        code_text=code_run_data.code_content,
        thought_chain=code_run_data.thought_chain or (llm_response.thought_chain if llm_response else ""),
        outputs=final_output,
        success=stop_type
        in [
            ExecStopType.NORMAL,
            ExecStopType.AGENT,
            ExecStopType.MULTIMODAL_AGENT,
        ],  # AGENT 状态也视为成功
        stop_type=stop_type,
        use_model=(llm_response and llm_response.use_model) or "",
        exec_time_ms=exec_time,
        generation_time_ms=generation_time_ms,
        total_time_ms=total_time,
        trigger_user_id=str(chat_message.sender_id or "0") if chat_message else "",
        trigger_user_name=chat_message.sender_name if chat_message else "System",
        extra_data=ext_data,
    )

    return final_output, output_text, stop_type.value


async def run_container_with_timeout(container: DockerContainer, timeout: int) -> Tuple[str, ExecStopType]:
    """运行容器并返回输出结果和退出类型"""
    try:
        task = asyncio.create_task(asyncio.wait_for(container.wait(), timeout=timeout))
        await asyncio.wait_for(task, timeout=timeout)
        outputs = await container.log(stdout=True, stderr=True)
        await container.delete()
        logger.info(f"容器 {container.id} 运行结束退出")

        # 检查输出中的结束标记来确定退出类型
        output_text = "".join(outputs).strip()
        stop_type = ExecStopType.ERROR  # 默认为错误退出

        # 移除所有结束标记并确定退出类型
        for _type, end_flag in CODE_RUN_END_FLAGS.items():
            if end_flag in output_text:
                stop_type = _type
                output_text = output_text.replace(end_flag, "").strip()
                break

    except asyncio.TimeoutError:
        logger.warning(f"容器 {container.id} 运行超过 {timeout} 秒，强制停止容器")
        outputs = await container.log(stdout=True, stderr=True)
        outputs.append(f"# This container has been killed because it exceeded the {timeout} seconds limit.")
        await container.kill()
        await container.delete()
        output_text = "".join(outputs).strip()
        # 移除所有可能的结束标记
        for end_flag in CODE_RUN_END_FLAGS.values():
            output_text = output_text.replace(end_flag, "").strip()
        return output_text, ExecStopType.TIMEOUT
    else:
        return output_text, stop_type


async def cleanup_sandbox_containers():
    """清理所有沙盒容器"""
    docker = aiodocker.Docker()
    try:
        containers = await docker.containers.list(all=True)
        for container in containers:
            container_info = await container.show()
            if IMAGE_NAME in container_info["Name"]:
                await container.kill()
                await container.delete()
                logger.info(f"已清理容器 {container_info['Name']}")
    finally:
        await docker.close()
