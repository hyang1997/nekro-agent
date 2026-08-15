"""
# 自拍 (Selfie)

让 AI 以「自己」的身份拍一张自拍照，而不是画一张无关的图。

## 设计思路

本插件**不拼提示词**。角色外观（`APPEARANCE`）通过提示注入直接交给 AI，由 AI 自己
写出完整的英文提示词——和内置 `draw` 插件效果好的原因一样：本地 Krea2 用的是 T5
文本编码器，吃自然语言长句，不吃 `1girl` 这类 booru 标签，而 AI 天生就会写前者。

插件只负责三件机械的事：锁种子（保证每次是同一张脸）、竖构图、把图发出去。

## 主要功能

- **自拍**: AI 调用 `take_selfie(prompt)`，生成并发送图片。
- **外观可改**: 改配置里的 `APPEARANCE` 即可换角色，无需改代码；可在频道级单独覆盖。
- **一致性**: `LOCK_FACE` 打开时使用固定种子，同一外观描述下长相基本稳定。

## 配置说明

- **绘图模型组**: 复用系统配置里的绘图模型组（默认 `comfy-krea2`，即本地 ComfyUI）。
- **外观描述**: 英文自然语言短句。这段会注入到 AI 的上下文里，AI 知道自己长什么样。

注意：本地 Krea2 工作流是蒸馏模型（cfg=1）且负面分支被 `ConditioningZeroOut` 清零，
因此**负面提示词无效**，插件不提供该配置项。
"""

import random

from httpx import AsyncClient, Timeout
from pydantic import Field

from nekro_agent.api import core, i18n
from nekro_agent.api.plugin import (
    ConfigBase,
    ExtraField,
    NekroPlugin,
    SandboxMethodType,
)
from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core.config import config as global_config

plugin = NekroPlugin(
    name="自拍",
    module_name="selfie",
    description="以固定人设外观拍一张自拍照",
    version="0.2.0",
    author="Hao",
    url="",
    i18n_name=i18n.i18n_text(zh_CN="自拍", en_US="Selfie"),
    i18n_description=i18n.i18n_text(
        zh_CN="以固定人设外观拍一张自拍照",
        en_US="Take a selfie with a fixed character appearance",
    ),
    allow_sleep=True,
    sleep_brief="提供以自身人设外观自拍的能力。仅在用户要求看照片/自拍时激活。",
)


@plugin.mount_config()
class SelfieConfig(ConfigBase):
    """自拍配置"""

    DRAW_MODEL_GROUP: str = Field(
        default="comfy-krea2",
        title="绘图模型组",
        description="使用的绘图模型组，可在 `系统配置` -> `模型组` 选项卡配置",
        json_schema_extra=ExtraField(
            ref_model_groups=True,
            required=True,
            model_type="draw",
            i18n_title=i18n.i18n_text(zh_CN="绘图模型组", en_US="Drawing Model Group"),
        ).model_dump(),
    )
    APPEARANCE: str = Field(
        default=(
            "a cute catgirl with long wavy pink hair, fluffy pink cat ears, bright "
            "amber eyes, fair skin and a slender build"
        ),
        title="角色外观",
        description=(
            "角色长相。必须是英文，且要用**自然语言短句**而不是 booru 标签"
            "（本地 Krea2 是 T5 编码器，`1girl` 这类标签几乎无效）。"
            "这段会注入到 AI 上下文，由 AI 自己写进提示词。改这里就能换角色；"
            "可在频道级单独覆盖，让不同人设各有各的长相。"
        ),
        json_schema_extra=ExtraField(
            is_textarea=True,
            overridable=True,
            i18n_title=i18n.i18n_text(zh_CN="角色外观", en_US="Character Appearance"),
        ).model_dump(),
    )
    SIZE: str = Field(
        default="768x1344",
        title="图片尺寸",
        description="竖构图更像自拍。768x1344 是本地 Krea2 工作流 1344x768 的竖版转置，像素量相同。",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="图片尺寸", en_US="Image Size"),
        ).model_dump(),
    )
    LOCK_FACE: bool = Field(
        default=True,
        title="锁定长相",
        description="打开后所有自拍共用同一个种子，人物长相基本稳定；关闭则每次随机。",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="锁定长相", en_US="Lock Face"),
        ).model_dump(),
    )
    FACE_SEED: int = Field(
        default=776402231,
        title="长相种子",
        description="`锁定长相` 打开时使用的种子。对当前长相不满意就换一个数字重抽。",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="长相种子", en_US="Face Seed"),
        ).model_dump(),
    )
    STEPS: int = Field(
        default=8,
        title="推理步数",
        description="本地 Krea2 Muse 是蒸馏模型，8 步即可。注意：桥接层默认忽略此值，除非设置 RESPECT_CLIENT_SAMPLER=1。",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="推理步数", en_US="Inference Steps"),
        ).model_dump(),
    )
    GUIDANCE_SCALE: float = Field(
        default=1.0,
        title="引导强度",
        description="蒸馏模型用 1.0；普通模型用 7.5 左右。同样默认被桥接层忽略。",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="引导强度", en_US="Guidance Scale"),
        ).model_dump(),
    )
    TIMEOUT: int = Field(
        default=300,
        title="超时时间",
        description="单位: 秒",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="超时时间", en_US="Timeout"),
        ).model_dump(),
    )


config: SelfieConfig = plugin.get_config(SelfieConfig)


@plugin.mount_prompt_inject_method(name="selfie_appearance")
async def inject_appearance(_ctx: AgentCtx) -> str:
    """把角色外观交给 AI，让它自己写提示词。"""
    appearance = config.APPEARANCE.strip()
    if not appearance:
        return ""
    return (
        "Your own physical appearance, for when you take a selfie with `take_selfie`: "
        f"{appearance}. "
        "Always write this appearance into the selfie prompt so every photo of you "
        "shows the same person."
    )


async def _generate(prompt: str, seed: int) -> str:
    """调用绘图模型组的 images/generations 接口，返回图片 URL。

    没有复用内置 draw 插件的实现，因为它每次都自己随机种子且不对外暴露，
    锁不住长相——而长相稳定正是自拍的意义所在。
    """
    group_name = config.DRAW_MODEL_GROUP
    if group_name not in global_config.MODEL_GROUPS:
        raise Exception(
            f"[Selfie] 绘图模型组 `{group_name}` 未配置。这不是临时错误，重试无用——"
            f"请把这个情况告诉用户，让其在 `系统配置` -> `模型组` 里检查配置。",
        )

    model_group = global_config.MODEL_GROUPS[group_name]

    json_data = {
        "model": model_group.CHAT_MODEL,
        "prompt": prompt,
        "image_size": config.SIZE,
        "batch_size": 1,
        "seed": seed,
        "num_inference_steps": config.STEPS,
        "guidance_scale": config.GUIDANCE_SCALE,
    }

    async with AsyncClient() as client:
        response = await client.post(
            f"{model_group.BASE_URL}/images/generations",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {model_group.API_KEY}",
            },
            json=json_data,
            timeout=Timeout(read=config.TIMEOUT, write=config.TIMEOUT, connect=10, pool=10),
        )

    response.raise_for_status()
    data = response.json()
    items = data.get("data") or []
    url = items[0].get("url") if items and isinstance(items[0], dict) else None
    if not url:
        core.logger.error(f"[Selfie] 绘图响应中没有图片: {data}")
        raise Exception("No image returned by the drawing model. Try adjusting the description.")
    return url


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="自拍",
    description="以自己的人设外观拍一张自拍照并发送",
)
async def take_selfie(_ctx: AgentCtx, prompt: str, send: bool = True) -> str:
    """Take a selfie of YOURSELF and send it to the chat.

    Write the whole prompt yourself, in ENGLISH, as flowing natural language --
    NOT booru tags. The local model uses a T5 text encoder, so descriptive
    sentences work well and tags like "1girl" do almost nothing.

    Include, roughly in this order:
      1. that it is a selfie, and the framing (arm's length, phone front camera,
         upper body, slight high angle -- whatever fits)
      2. YOUR appearance, exactly as given to you in your context
      3. what you are wearing
      4. your pose and expression
      5. where you are / the background
      6. lighting, mood, and art style

    The photo is portrait orientation and your face is seed-locked, so the same
    person shows up every time as long as you keep the appearance consistent.

    Args:
        prompt (str): The complete image description, in English prose.
        send (bool): Send the photo to the current chat. Default True.

    Returns:
        str: Sandbox path of the generated photo.

    Examples:
        take_selfie("an anime style selfie taken at arm's length with a phone front camera, a cute catgirl with long wavy pink hair, fluffy pink cat ears and bright amber eyes, wearing an oversized cream hoodie, smiling softly and looking straight at the camera, sitting by a cafe window with afternoon sunlight on her face, warm cozy lighting, detailed anime illustration, high quality")

        # Generate without sending, so you can decide whether to use it
        take_selfie("an anime style selfie ... at the gym mirror, grinning", send=False)
    """
    seed = config.FACE_SEED if config.LOCK_FACE else random.randint(0, 2**31 - 1)

    core.logger.info(f"[Selfie] seed={seed} prompt={prompt}")

    url = await _generate(prompt, seed)
    sandbox_path = await _ctx.fs.mixed_forward_file(url, file_name="selfie.png")

    if send:
        await _ctx.send_image(sandbox_path)

    return sandbox_path


@plugin.mount_cleanup_method()
async def clean_up():
    """清理插件"""
