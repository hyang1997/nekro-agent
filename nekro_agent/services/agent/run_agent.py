import datetime
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from nekro_agent.core.config import CoreConfig, ModelConfigGroup
from nekro_agent.core.logger import get_sub_logger
from nekro_agent.core.os_env import PROMPT_ERROR_LOG_DIR, PROMPT_LOG_DIR
from nekro_agent.models.db_chat_channel import DBChatChannel
from nekro_agent.models.db_chat_message import DBChatMessage
from nekro_agent.models.db_exec_code import ExecStopType
from nekro_agent.schemas.agent_ctx import AgentCtx
from nekro_agent.schemas.chat_message import ChatMessage
from nekro_agent.services.plugin.collector import plugin_collector
from nekro_agent.services.plugin.prompt_activation import build_plugin_activation_rules
from nekro_agent.services.sandbox.runner import limited_run_code

from .creator import OpenAIChatMessage
from .openai import OpenAIResponse, gen_openai_chat_response
from .resolver import ParsedCodeRunData, parse_chat_response
from .templates.compiler import PromptCompiler
from .templates.history import render_history_data
from .templates.plugin import render_plugins_prompt

# 使用deque保存最近100条错误日志路径

logger = get_sub_logger("agent_runtime")
RECENT_ERR_LOGS = deque(maxlen=100)


def _summarize_runtime_text(text: str, limit: int = 160) -> str:
    compact = " ".join(text.strip().split())
    if not compact:
        return ""
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


class AllLLMRequestsFailedError(ValueError):
    """All LLM API retries are exhausted for a single agent request."""


# 补发指令。措辞是刻意具体的：DeepSeek 在同类修复里 A/B 过，不给明确指令时
# 模型的收尾"高方差，甚至自信地编造出文件级细节"。所以这里既说清要做什么，
# 也明确禁止编造没真正拿到的结果。
_WRAPUP_INSTRUCTION = (
    "[System] Your code ran successfully but sent nothing to the user, and this turn was "
    "triggered by their message -- so right now they are looking at silence.\n"
    "{output_block}"
    "Reply to them now using send_msg_text. Base it strictly on the output above and what you "
    "already know; do NOT invent results you did not actually obtain. If there is no useful "
    "output and you cannot answer, say plainly what you tried and what you still need."
)


# 报错到达模型时，附上"下一步该做什么"。写在系统提示里的静态说明到不了这个决策点：
# 失败发生在半途，而这条建议恰好出现在模型必须决定怎么重试的那一刻。
# 顺序敏感——先命中的先用，所以更具体的异常放前面（IndentationError 必须在 SyntaxError 之前，
# 因为它是 SyntaxError 的子类，traceback 里两个名字都可能出现）。
_EXCEPTION_SUGGESTIONS: List[Tuple[str, str]] = [
    (
        "IndentationError",
        "Re-emit the ENTIRE script with consistent indentation. Do not wrap it in markdown fences, "
        "headings, or prose — anything that is not Python breaks the parse.",
    ),
    (
        "SyntaxError",
        "You are prohibited from adding anything other than the content of the code that might break the "
        "syntax of the code. Please ensure your output specification and try again",
    ),
    (
        "ModuleNotFoundError",
        "That package is NOT installed in the sandbox. Either install and import it at runtime with "
        'dynamic_importer("<package>"), or solve the task using the dependencies listed in your prompt.',
    ),
    (
        "ImportError",
        "Check the import name against the package name — they often differ. If the package is genuinely "
        'absent, install it at runtime with dynamic_importer("<package>").',
    ),
    (
        "NameError",
        "You used a name that does not exist. Predefined methods are ONLY the ones declared in the plugin "
        "blocks above; you must not invent method names, and you must not import or redefine them. Use a "
        "method that actually exists, or do the work in plain Python.",
    ),
    (
        "FileNotFoundError",
        "Files must live under ./shared/, and that directory is cleared after a period of inactivity, so a "
        "path from an earlier conversation may already be gone. Confirm before using it: "
        "print(os.listdir('./shared')); exit(9)",
    ),
    (
        "TypeError",
        "Re-read the signature of the predefined method in the plugin block above and pass exactly the "
        "arguments it declares. Do not guess parameter names, order, or types.",
    ),
]

# 按退出类型给的建议，与 traceback 文本无关
_STOP_TYPE_SUGGESTIONS: Dict[ExecStopType, str] = {
    ExecStopType.TIMEOUT: (
        "The sandbox is killed at {timeout} seconds. Do not sleep, poll, or retry in a loop inside the "
        "script — split the work across iterations and fetch less per run."
    ),
}


def _resolve_suggestion(sandbox_output: str, stop_type: ExecStopType, timeout: int) -> str:
    """给本轮失败挑一条可执行的恢复建议，没有合适的就返回空串"""
    if stop_type in _STOP_TYPE_SUGGESTIONS:
        return _STOP_TYPE_SUGGESTIONS[stop_type].format(timeout=timeout)
    for marker, suggestion in _EXCEPTION_SUGGESTIONS:
        if marker in sandbox_output:
            return suggestion
    return ""


async def _bot_replied_since(chat_key: str, bot_nickname: str, since_ts: float) -> bool:
    """本轮里机器人是否真的对用户说了话。

    只认以人设名义发出的消息：系统提示以 "SYSTEM" 落库，不算对用户的回复。
    """
    return await DBChatMessage.filter(
        chat_key=chat_key,
        sender_nickname=bot_nickname,
        send_timestamp__gte=int(since_ts),
    ).exists()


async def run_agent(
    chat_key: str,
    chat_message: Optional[ChatMessage] = None,
    ctx: Optional[AgentCtx] = None,
):
    # 获取当前聊天频道的有效配置
    one_time_code = os.urandom(4).hex()
    db_chat_channel: DBChatChannel
    logger.debug(f"[run_agent] {chat_key} | 开始准备上下文")
    if ctx:
        if ctx.db_chat_channel:
            db_chat_channel = ctx.db_chat_channel
        else:
            db_chat_channel = await DBChatChannel.get(chat_key=chat_key)
            ctx = AgentCtx.create_by_db_chat_channel(db_chat_channel=db_chat_channel)
    else:
        db_chat_channel = await DBChatChannel.get(chat_key=chat_key)
        ctx = AgentCtx.create_by_db_chat_channel(db_chat_channel=db_chat_channel)

    config = await db_chat_channel.get_effective_config()
    preset = await db_chat_channel.get_preset()
    adapter_dialog_examples = await ctx.adapter.set_dialog_example()
    adapter_jinja_env = await ctx.adapter.get_jinja_env()
    self_info = await ctx.adapter.get_self_info()
    logger.debug(f"[run_agent] {chat_key} | 适配器信息就绪，开始渲染插件 prompt")

    # 获取当前使用的模型组
    used_model_group: ModelConfigGroup = config.MODEL_GROUPS[config.USE_MODEL_GROUP]
    activation_plugin = plugin_collector.get_plugin_by_module_name("plugin_activation")
    activation_enabled = bool(activation_plugin and activation_plugin.is_enabled)

    rendered_plugins = await render_plugins_prompt(
        plugin_collector.get_all_active_plugins(),
        ctx,
        activation_enabled=activation_enabled,
    )
    adapter_runtime_prompt = await ctx.adapter.render_runtime_prompt()
    runtime_prompts = [
        prompt for prompt in (adapter_runtime_prompt, rendered_plugins.runtime_prompt) if prompt.strip()
    ]
    runtime_prompt = "\n\n".join(runtime_prompts)
    logger.debug(f"[run_agent] {chat_key} | 插件 prompt 渲染完成，开始构建 system prompt")

    from nekro_agent.services.system_broadcast import AgentRuntimeStatusEvent, publish_system_event

    started_at = int(time.time() * 1000)
    iteration_total = config.AI_SCRIPT_MAX_RETRY_TIMES + 1
    llm_retry_total = config.AI_CHAT_LLM_API_MAX_RETRIES

    async def publish_runtime_state(
        *,
        phase: str,
        iteration_index: int,
        llm_retry_index: int = 1,
        model_name: Optional[str] = None,
        sandbox_stop_type: Optional[int] = None,
        error_summary: Optional[str] = None,
    ) -> None:
        try:
            await publish_system_event(
                AgentRuntimeStatusEvent(
                    chat_key=chat_key,
                    active=True,
                    channel_name=db_chat_channel.channel_name,
                    chat_type=db_chat_channel.channel_type,
                    preset_id=getattr(preset, "id", None),
                    preset_name=getattr(preset, "name", None),
                    started_at=started_at,
                    updated_at=int(time.time() * 1000),
                    phase=phase,
                    iteration_index=iteration_index,
                    iteration_total=iteration_total,
                    llm_retry_index=llm_retry_index,
                    llm_retry_total=llm_retry_total,
                    sandbox_stop_type=sandbox_stop_type,
                    model_name=model_name,
                    error_summary=error_summary,
                )
            )
        except Exception as e:
            logger.debug(f"[run_agent] {chat_key} | 运行阶段广播失败: {e}")

    prompt_compiler = PromptCompiler(
        platform_name=self_info.platform_name,
        bot_platform_id=self_info.user_id,
        chat_preset=preset.content,
        plugins_prompt=rendered_plugins.system_prompt,
        plugins_runtime_prompt=runtime_prompt,
        plugin_activation_rules=build_plugin_activation_rules() if activation_enabled else "",
        enable_cot=used_model_group.ENABLE_COT,
        chat_key_rules="\n".join(f"- {r}" for r in db_chat_channel.adapter.chat_key_rules),
        enable_at=db_chat_channel.adapter.config.SESSION_ENABLE_AT,
    )

    messages = [prompt_compiler.render_system_message()]
    messages.extend(prompt_compiler.render_practice_messages(adapter_dialog_examples, adapter_jinja_env))
    messages.append(
        await prompt_compiler.render_history_message(
            chat_key=chat_key,
            db_chat_channel=db_chat_channel,
            one_time_code=one_time_code,
            config=config,
            model_group=used_model_group,
        ),
    )

    logger.debug(f"[run_agent] {chat_key} | 历史记录渲染完成，发送 LLM 请求 (model={used_model_group.CHAT_MODEL})")
    history_render_until_time = time.time()
    llm_retry_errors: list[str] = []
    try:
        llm_response, used_model_group, llm_retry_errors = await send_agent_request(
            messages=messages,
            config=config,
            chat_key=chat_key,
            on_llm_retry=lambda retry_index, retry_total, model_name, error_summary: publish_runtime_state(
                phase="llm_retrying",
                iteration_index=1,
                llm_retry_index=retry_index,
                model_name=model_name,
                error_summary=error_summary,
            ),
            on_llm_attempt=lambda retry_index, _retry_total, model_name: publish_runtime_state(
                phase="llm_generating",
                iteration_index=1,
                llm_retry_index=retry_index,
                model_name=model_name,
            ),
        )
    except AllLLMRequestsFailedError as e:
        await publish_runtime_state(
            phase="failed",
            iteration_index=1,
            llm_retry_index=llm_retry_total,
            error_summary=_summarize_runtime_text(str(e)),
        )
        raise
    logger.debug(f"[run_agent] {chat_key} | LLM 请求完成，开始解析响应")
    parsed_code_data: ParsedCodeRunData = parse_chat_response(llm_response.response_content)

    # 循环体每轮都会重置这两个值；此处预置一次，避免迭代次数被配置为 0 时
    # 循环体从未执行，而循环后的收尾逻辑引用到未绑定的名字
    sandbox_output = ""
    stop_type = ExecStopType.NORMAL

    turn_start_ts = time.time()
    wrapup_used = False  # 补发只做一次，避免和模型来回拉扯

    for i in range(config.AI_SCRIPT_MAX_RETRY_TIMES):
        addition_prompt_message: List[OpenAIChatMessage] = []
        sandbox_output = ""
        raw_output = ""
        current_iteration = i + 1
        if one_time_code in parsed_code_data.code_content:
            stop_type = ExecStopType.SECURITY
        else:
            await publish_runtime_state(
                phase="sandbox_running",
                iteration_index=current_iteration,
                model_name=llm_response.use_model,
            )
            sandbox_output, raw_output, stop_type_value = await limited_run_code(
                code_run_data=parsed_code_data,
                from_chat_key=chat_key,
                chat_message=chat_message,
                llm_response=llm_response,
                ctx=ctx,
                llm_retry_errors=llm_retry_errors,
            )
            stop_type = ExecStopType(stop_type_value)

        # "代码跑通了" 和 "用户收到了回复" 在这个框架里是两件独立的事：脚本 exit 0
        # 本轮就结束，忘了 send_msg_text 也照样算成功，用户对着空气等。由用户消息触发
        # 却一句话没说的回合，几乎都是模型以为还能再来一轮（典型是 print(...) 查完就
        # 结束）。这里补一轮，把 stdout 还给它并明确要求现在回复。
        needs_wrapup = False
        if stop_type == ExecStopType.NORMAL:
            if (
                chat_message is not None
                and not wrapup_used
                # 本轮产出的代码要到下一轮才执行，所以最后一轮补发等于白补
                and i < config.AI_SCRIPT_MAX_RETRY_TIMES - 1
                and not await _bot_replied_since(chat_key, preset.name, turn_start_ts)
            ):
                needs_wrapup = True
                wrapup_used = True
                logger.warning(
                    f"[run_agent] {chat_key} | 本轮执行成功但没有回复用户，补发一轮收尾提示",
                )
            else:
                await publish_runtime_state(
                    phase="completed",
                    iteration_index=current_iteration,
                    model_name=llm_response.use_model,
                )
                return

        await publish_runtime_state(
            phase="sandbox_stopped",
            iteration_index=current_iteration,
            model_name=llm_response.use_model,
            sandbox_stop_type=stop_type.value,
            error_summary=_summarize_runtime_text(sandbox_output),
        )

        # 添加 AI 回复的原始内容到上下文
        addition_prompt_message.append(OpenAIChatMessage.from_text("assistant", llm_response.response_content))

        msg: OpenAIChatMessage = OpenAIChatMessage.create_empty("user")  # 待添加到迭代上下文的用户消息

        # Agent 类型的迭代对话
        if stop_type == ExecStopType.AGENT:
            msg = msg.extend(
                OpenAIChatMessage.from_text(
                    "user",
                    f"[Agent Method Response] {sandbox_output}\nPlease continue based on this agent response. Attention: the code after the agent method is NOT EXECUTED!",
                ),
            )

        # 多模态类型的迭代对话
        elif stop_type == ExecStopType.MULTIMODAL_AGENT:
            multimodal_agent_result = json.loads(raw_output.split("<AGENT_RESULT>")[1].split("</AGENT_RESULT>")[0])
            if isinstance(multimodal_agent_result, list):
                msg = msg.extend(OpenAIChatMessage("user", multimodal_agent_result))
            elif isinstance(multimodal_agent_result, str):
                msg = msg.extend(OpenAIChatMessage.from_text("user", multimodal_agent_result))
            elif isinstance(multimodal_agent_result, dict):
                msg = msg.extend(OpenAIChatMessage(**multimodal_agent_result))
            else:
                raise ValueError(f"Multimodal agent result is not a list or string: {multimodal_agent_result}")
            msg = msg.extend(OpenAIChatMessage.from_text("user", "Attention: the code AFTER THE AGENT METHOD is NOT EXECUTED!"))

        # 异常类型的迭代对话
        exception_reason_map: Dict[ExecStopType, str] = {
            ExecStopType.TIMEOUT: "Sandbox exited due to timeout",
            ExecStopType.ERROR: "Sandbox exited due to error occurred",
            ExecStopType.MANUAL: "Sandbox exited due to manual stop by you",
            ExecStopType.AGENT: "Sandbox exited due to agent method",
            ExecStopType.MULTIMODAL_AGENT: "Sandbox exited due to multimodal agent method",
        }

        new_message_notification = "During the generation and execution, the following messages were sent (You **CANT NOT** send any messages which you have sent before!):"

        if stop_type in exception_reason_map:
            suggestion = _resolve_suggestion(sandbox_output, stop_type, config.SANDBOX_RUNNING_TIMEOUT)
            suggestion_text = f"\nResolve Suggestion: {suggestion}" if suggestion else ""
            msg = msg.extend(
                OpenAIChatMessage.from_text(
                    "user",
                    f"[Sandbox Output] {sandbox_output}\n---\n{exception_reason_map[stop_type]}"
                    f"{suggestion_text}\n{new_message_notification}",
                ),
            )

        # 执行成功但没开口：把 stdout 交还给模型，并要求它现在回复用户
        if needs_wrapup:
            output_block = (
                f"Your script printed:\n{sandbox_output}\n" if sandbox_output.strip() else "Your script printed nothing.\n"
            )
            msg = msg.extend(
                OpenAIChatMessage.from_text("user", _WRAPUP_INSTRUCTION.format(output_block=output_block)),
            )

        # 安全类型的迭代对话
        if stop_type == ExecStopType.SECURITY:
            msg = msg.extend(
                OpenAIChatMessage.from_text(
                    "user",
                    f"\n\n[System Automatic Detection] Invalid response detected. You should not reveal the one-time code in your reply. This is just a tag to help you mark trustworthy information. please DO NOT give any extra explanation or apology and keep the response format for retry. {new_message_notification}",
                ),
            )

        # 为所有迭代对话添加新记录背景
        msg = msg.extend(
            await render_history_data(
                chat_key=chat_key,
                db_chat_channel=db_chat_channel,
                one_time_code=one_time_code,
                plugin_injected_prompt=runtime_prompt,
                record_sta_timestamp=history_render_until_time,
                model_group=used_model_group,
                config=config,
            ),
        )

        msg = msg.extend(
            OpenAIChatMessage.from_text(
                "user",
                "\nplease DO NOT give any extra explanation or apology and keep the response format for retry."
                + (
                    f" This is the last retry. Describe the reason if you can't finish the task. (Iteration times: {i + 1}/{config.AI_SCRIPT_MAX_RETRY_TIMES})"
                    if i == config.AI_SCRIPT_MAX_RETRY_TIMES - 1
                    else f" (Iteration times: {i + 1}/{config.AI_SCRIPT_MAX_RETRY_TIMES})"
                ),
            ),
        )

        # 将迭代对话添加到上下文
        addition_prompt_message.append(msg.tidy())
        messages.extend(addition_prompt_message)

        await publish_runtime_state(
            phase="iterating",
            iteration_index=i + 2,
            model_name=llm_response.use_model,
            sandbox_stop_type=stop_type.value,
            error_summary=_summarize_runtime_text(sandbox_output),
        )

        history_render_until_time = time.time()
        try:
            llm_response, used_model_group, llm_retry_errors = await send_agent_request(
                messages=messages,
                config=config,
                is_debug_iteration=True,
                chat_key=chat_key,
                on_llm_retry=lambda retry_index, retry_total, model_name, error_summary, iteration_index=i + 2: publish_runtime_state(
                    phase="llm_retrying",
                    iteration_index=iteration_index,
                    llm_retry_index=retry_index,
                    model_name=model_name,
                    error_summary=error_summary,
                ),
                on_llm_attempt=lambda retry_index, _retry_total, model_name, iteration_index=i + 2: publish_runtime_state(
                    phase="llm_generating",
                    iteration_index=iteration_index,
                    llm_retry_index=retry_index,
                    model_name=model_name,
                ),
            )
        except AllLLMRequestsFailedError as e:
            await publish_runtime_state(
                phase="failed",
                iteration_index=i + 2,
                llm_retry_index=llm_retry_total,
                error_summary=_summarize_runtime_text(str(e)),
            )
            raise
        parsed_code_data = parse_chat_response(llm_response.response_content)

    # 迭代次数用尽。此前这里没有任何收尾：函数直接返回 None，既不记日志也不
    # 发消息，用户看到的就是机器人对着一条消息毫无反应，无从判断是没收到、
    # 还是在想、还是已经放弃。至少要让失败可见。
    logger.error(
        f"[run_agent] {chat_key} | 连续 {config.AI_SCRIPT_MAX_RETRY_TIMES} 次迭代仍未成功执行，放弃本轮响应"
        f" | 最后一次: {stop_type.name} {_summarize_runtime_text(sandbox_output)}",
    )
    await publish_runtime_state(
        phase="failed",
        iteration_index=config.AI_SCRIPT_MAX_RETRY_TIMES,
        model_name=llm_response.use_model,
        sandbox_stop_type=stop_type.value,
        error_summary=_summarize_runtime_text(sandbox_output),
    )

    if config.AI_NOTIFY_ON_SCRIPT_FAILURE:
        try:
            await ctx.send_text(
                f"[系统] 这轮响应连续失败 {config.AI_SCRIPT_MAX_RETRY_TIMES} 次，已放弃。"
                f"最后一次错误：{_summarize_runtime_text(sandbox_output)}",
                record=False,
            )
        except Exception as e:
            # 失败通知本身失败了也不能再把异常抛给调用方
            logger.error(f"[run_agent] {chat_key} | 发送失败通知时出错: {e}")


async def send_agent_request(
    messages: List[OpenAIChatMessage],
    config: CoreConfig,
    is_debug_iteration: bool = False,
    chat_key: str = "",
    on_llm_attempt: Optional[Callable[[int, int, str], Awaitable[None]]] = None,
    on_llm_retry: Optional[Callable[[int, int, str, str], Awaitable[None]]] = None,
) -> Tuple[OpenAIResponse, ModelConfigGroup, list[str]]:
    model_group: ModelConfigGroup = (
        config.MODEL_GROUPS[config.DEBUG_MIGRATION_MODEL_GROUP]
        if is_debug_iteration and config.DEBUG_MIGRATION_MODEL_GROUP
        else config.MODEL_GROUPS[config.USE_MODEL_GROUP]
    )
    fallback_model_group: ModelConfigGroup = (
        config.MODEL_GROUPS[config.FALLBACK_MODEL_GROUP] if config.FALLBACK_MODEL_GROUP else model_group
    )

    if config.SAVE_PROMPTS_LOG:
        log_path = f"{PROMPT_LOG_DIR}/chat_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    else:
        log_path = None
    err_log_path = (
        f"{PROMPT_ERROR_LOG_DIR}/chat_err_{model_group.CHAT_MODEL}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    )

    used_model_group: ModelConfigGroup = model_group  # 记录实际使用的模型组
    retry_errors: list[str] = []

    for i in range(config.AI_CHAT_LLM_API_MAX_RETRIES):
        use_model_group: ModelConfigGroup = model_group if i < config.AI_CHAT_LLM_API_MAX_RETRIES - 1 else fallback_model_group
        retry_index = i + 1
        if on_llm_attempt is not None:
            await on_llm_attempt(retry_index, config.AI_CHAT_LLM_API_MAX_RETRIES, use_model_group.CHAT_MODEL)

        logger.info(
            f"[send_agent_request] {chat_key} | 发送 LLM 请求 model={use_model_group.CHAT_MODEL} retry={i}/{config.AI_CHAT_LLM_API_MAX_RETRIES}"
        )
        try:
            llm_response: OpenAIResponse = await gen_openai_chat_response(
                model=use_model_group.CHAT_MODEL,
                messages=messages,
                temperature=use_model_group.TEMPERATURE,
                top_p=use_model_group.TOP_P,
                top_k=use_model_group.TOP_K,
                frequency_penalty=use_model_group.FREQUENCY_PENALTY,
                presence_penalty=use_model_group.PRESENCE_PENALTY,
                extra_body=use_model_group.EXTRA_BODY,
                base_url=use_model_group.BASE_URL,
                api_key=use_model_group.API_KEY,
                stream_mode=config.AI_REQUEST_STREAM_MODE,
                proxy_url=use_model_group.CHAT_PROXY,
                max_wait_time=config.AI_GENERATE_TIMEOUT,
                first_token_timeout=config.AI_STREAM_FIRST_TOKEN_TIMEOUT,
                log_path=log_path,
                error_log_path=err_log_path,
            )
        except Exception as e:
            error_summary = _summarize_runtime_text(str(e))
            retry_errors.append(str(e))
            logger.error(
                f"LLM 请求失败: {e} ｜ 使用模型: {use_model_group.CHAT_MODEL} {'(fallback)' if i == config.AI_CHAT_LLM_API_MAX_RETRIES - 1 else ''}",
            )
            if on_llm_retry is not None:
                await on_llm_retry(retry_index, config.AI_CHAT_LLM_API_MAX_RETRIES, use_model_group.CHAT_MODEL, error_summary)
            # 避免重复添加，转换为Path对象并比较绝对路径
            err_log_path_obj = Path(err_log_path)
            if not any(str(log_path.absolute()) == str(err_log_path_obj.absolute()) for log_path in RECENT_ERR_LOGS):
                RECENT_ERR_LOGS.append(err_log_path_obj)
            continue
        else:
            used_model_group = use_model_group  # 记录成功使用的模型组
            break
    else:
        err_log = Path(f"{PROMPT_LOG_DIR}/chat_err_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log")
        err_log.parent.mkdir(parents=True, exist_ok=True)
        err_log.write_text(
            json.dumps([message.to_dict() for message in messages], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise AllLLMRequestsFailedError("所有 LLM 请求失败")

    return llm_response, used_model_group, retry_errors
