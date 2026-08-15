"""
# Strava 运动数据 (Strava Activities)

让 AI 直接读取 Strava 运动记录，而不必每次自己现写 HTTP 请求。

## 主要功能

- **最近活动**: 获取最近的跑步/骑行/游泳等活动列表。
- **活动详情**: 按 ID 获取单次活动的完整数据。
- **运动统计**: 获取年度/历史累计里程与时间。

## 配置说明

需要在插件配置中填入 Strava 应用的 `CLIENT_ID`、`CLIENT_SECRET` 和一个带
`activity:read_all` 授权范围的 `REFRESH_TOKEN`。

授权链接（把 CLIENT_ID 换成你自己的）：

    https://www.strava.com/oauth/authorize?client_id=<CLIENT_ID>
        &response_type=code&redirect_uri=http://localhost/exchange_token
        &approval_prompt=force&scope=activity:read_all

注意 scope 必须包含 `activity:read_all`，否则接口会返回 401
`activity:read_permission missing`——这是授权范围问题，任何重试都无法绕过。

Access token 由插件自动刷新并缓存；Strava 会轮换 refresh token，插件会把新的
refresh token 写回自己的存储，因此首次配置后一般无需再手动更新。
"""

import asyncio
import time
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from httpx import AsyncClient
from pydantic import Field

from nekro_agent.api import core, i18n
from nekro_agent.api.plugin import (
    ConfigBase,
    ExtraField,
    NekroPlugin,
    SandboxMethodType,
)
from nekro_agent.api.schemas import AgentCtx
from nekro_agent.services.message_service import message_service

plugin = NekroPlugin(
    name="Strava 运动数据",
    module_name="strava",
    description="读取 Strava 运动记录与统计数据",
    version="0.1.0",
    author="Hao",
    url="https://www.strava.com/",
    i18n_name=i18n.i18n_text(zh_CN="Strava 运动数据", en_US="Strava Activities"),
    i18n_description=i18n.i18n_text(
        zh_CN="读取 Strava 运动记录与统计数据",
        en_US="Read Strava activities and stats",
    ),
    allow_sleep=True,
    sleep_brief="用于查询用户的 Strava 运动记录（跑步/骑行等）、单次活动详情与累计统计。涉及运动、训练、里程时激活。",
)

TOKEN_URL = "https://www.strava.com/oauth/token"
API_BASE = "https://www.strava.com/api/v3"

# store keys
_K_ACCESS = "access_token"
_K_EXPIRES = "expires_at"
_K_REFRESH = "refresh_token"

SCOPE_HINT = (
    "[Strava] 授权范围不足：token 缺少 activity:read_all 权限。"
    "这不是临时错误，重试无用——需要用户重新授权。请把这个情况直接告诉用户，"
    "并让其使用带 scope=activity:read_all 的授权链接重新获取 refresh token。"
)


@plugin.mount_config()
class StravaConfig(ConfigBase):
    """Strava 配置"""

    CLIENT_ID: str = Field(
        default="",
        title="Strava Client ID",
        description="Strava 应用的 Client ID，<a href='https://www.strava.com/settings/api' target='_blank' rel='noopener noreferrer'>获取地址</a>",
        json_schema_extra=ExtraField(
            is_secret=True,
            i18n_title=i18n.i18n_text(zh_CN="Strava Client ID", en_US="Strava Client ID"),
        ).model_dump(),
    )
    CLIENT_SECRET: str = Field(
        default="",
        title="Strava Client Secret",
        description="Strava 应用的 Client Secret",
        json_schema_extra=ExtraField(
            is_secret=True,
            i18n_title=i18n.i18n_text(zh_CN="Strava Client Secret", en_US="Strava Client Secret"),
        ).model_dump(),
    )
    REFRESH_TOKEN: str = Field(
        default="",
        title="Strava Refresh Token",
        description="带 activity:read_all 授权范围的 refresh token（仅首次使用，之后由插件自动轮换）",
        json_schema_extra=ExtraField(
            is_secret=True,
            i18n_title=i18n.i18n_text(zh_CN="Strava Refresh Token", en_US="Strava Refresh Token"),
        ).model_dump(),
    )
    MAX_ACTIVITIES: int = Field(
        default=30,
        title="单次最大返回活动数",
        description="限制一次查询返回的活动条数，避免占用过多上下文",
    )
    WEBHOOK_VERIFY_TOKEN: str = Field(
        default="nekro-strava",
        title="Webhook 校验令牌",
        description="创建 Strava 订阅时使用的 verify_token，需与订阅请求中的一致",
        json_schema_extra=ExtraField(is_secret=True).model_dump(),
    )
    WEBHOOK_NOTIFY_CHAT_KEY: str = Field(
        default="",
        title="活动通知频道",
        description="收到 Strava 活动事件后推送到的会话 chat_key，留空则不推送",
    )
    WEBHOOK_NOTIFY_ON_UPDATE: bool = Field(
        default=False,
        title="活动更新时也通知",
        description="开启后，重命名/修改活动也会触发通知（Strava 会为编辑操作发送 update 事件）",
    )
    WEBHOOK_COMPARE_COUNT: int = Field(
        default=10,
        title="对比参考活动数",
        description="推送时拉取最近多少条同类型活动用于对比（0 表示不做对比）",
    )
    WEBHOOK_MAX_SPLITS: int = Field(
        default=15,
        title="最多附带分段数",
        description="推送时附带的每公里分段数量上限，避免长距离活动占满上下文",
    )
    WEBHOOK_PROMPT: str = Field(
        default=(
            "以上是用户刚完成的一次运动的完整数据。请像一个懂训练的朋友那样，主动开口聊这次运动：\n"
            "1. 先看分段配速是正split还是负split，是否掉速，据此判断配速控制和体力分配；\n"
            "2. 结合心率看强度（是轻松跑/节奏跑/间歇），有心率漂移就指出来；\n"
            "3. 和最近同类型活动对比，指出这次是进步、退步还是正常波动，用具体数字说话；\n"
            "4. 给一条具体、可执行的建议，不要泛泛而谈。\n"
            "用你自己的口吻自然地说，不要罗列数据表格，不要复述原始数字堆砌。"
        ),
        title="分析指令",
        description="附加在运动数据后面、用于引导 AI 如何点评这次运动的提示词",
    )


config: StravaConfig = plugin.get_config(StravaConfig)


# --------------------------------------------------------------- auth


async def _get_access_token() -> str:
    """Return a valid access token, refreshing (and persisting) when needed.

    Raises RuntimeError with a user-facing message when not configured.
    """
    if not config.CLIENT_ID or not config.CLIENT_SECRET:
        raise RuntimeError("[Strava] 未配置 CLIENT_ID / CLIENT_SECRET，请在插件配置中填写")

    cached = await plugin.store.get(store_key=_K_ACCESS)
    expires_raw = await plugin.store.get(store_key=_K_EXPIRES)
    if cached and expires_raw:
        try:
            # refresh a minute early to avoid racing the expiry
            if int(float(expires_raw)) - 60 > time.time():
                return cached
        except ValueError:
            pass

    # stored refresh token wins -- Strava rotates them on every refresh
    refresh = await plugin.store.get(store_key=_K_REFRESH) or config.REFRESH_TOKEN
    if not refresh:
        raise RuntimeError("[Strava] 未配置 REFRESH_TOKEN，请在插件配置中填写")

    async with AsyncClient() as cli:
        resp = await cli.post(
            TOKEN_URL,
            data={
                "client_id": config.CLIENT_ID,
                "client_secret": config.CLIENT_SECRET,
                "grant_type": "refresh_token",
                "refresh_token": refresh,
            },
            timeout=30,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"[Strava] 刷新 token 失败 ({resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    access = data.get("access_token", "")
    if not access:
        raise RuntimeError(f"[Strava] 刷新响应中没有 access_token: {resp.text[:300]}")

    await plugin.store.set(store_key=_K_ACCESS, value=access)
    await plugin.store.set(store_key=_K_EXPIRES, value=str(data.get("expires_at", 0)))
    if data.get("refresh_token"):
        await plugin.store.set(store_key=_K_REFRESH, value=data["refresh_token"])

    return access


async def _api_get(path: str, params: Optional[dict] = None):
    """GET against the Strava API. Returns parsed JSON, or raises RuntimeError
    with a message that is already meaningful to the agent."""
    token = await _get_access_token()
    async with AsyncClient() as cli:
        resp = await cli.get(
            f"{API_BASE}{path}",
            headers={"Authorization": f"Bearer {token}"},
            params=params or {},
            timeout=30,
        )

    if resp.status_code == 401:
        body = resp.text
        if "activity:read" in body:
            raise RuntimeError(SCOPE_HINT)
        raise RuntimeError(f"[Strava] 认证失败 (401)，token 可能已失效: {body[:300]}")
    if resp.status_code == 429:
        raise RuntimeError("[Strava] 触发接口限流 (429)，请稍后再试，不要立即重试")
    if resp.status_code != 200:
        raise RuntimeError(f"[Strava] 请求失败 ({resp.status_code}): {resp.text[:300]}")

    return resp.json()


def _fmt_activity(a: dict) -> str:
    dist_km = (a.get("distance") or 0) / 1000
    moving_min = (a.get("moving_time") or 0) / 60
    pace = f"{moving_min / dist_km:.2f} min/km" if dist_km > 0.05 else "-"
    return (
        f"- [{a.get('id')}] {a.get('name', '(无标题)')} | {a.get('type', '?')} | "
        f"{dist_km:.2f} km | {moving_min:.0f} min | 配速 {pace} | "
        f"爬升 {a.get('total_elevation_gain', 0):.0f} m | {a.get('start_date_local', '')}"
    )


# --------------------------------------------------------------- methods


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="获取Strava最近活动",
    description="获取用户最近的 Strava 运动记录列表",
)
async def get_recent_activities(_ctx: AgentCtx, limit: int = 10) -> str:
    """获取用户最近的 Strava 运动记录

    Args:
        limit (int): 返回的活动条数，默认 10

    Returns:
        str: 活动列表文本，每行一条活动摘要
    """
    limit = max(1, min(limit, config.MAX_ACTIVITIES))
    try:
        data = await _api_get("/athlete/activities", {"per_page": limit})
    except RuntimeError as e:
        return str(e)
    except Exception as e:
        core.logger.exception("Strava 获取活动失败")
        return f"[Strava] 获取活动失败: {e!s}"

    if not data:
        return "[Strava] 最近没有任何活动记录"

    return f"[Strava 最近 {len(data)} 条活动]\n" + "\n".join(_fmt_activity(a) for a in data)


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="获取Strava活动详情",
    description="按活动 ID 获取单次运动的详细数据",
)
async def get_activity_detail(_ctx: AgentCtx, activity_id: int) -> str:
    """按 ID 获取单次 Strava 活动的详细数据

    Args:
        activity_id (int): 活动 ID，可从最近活动列表中获得

    Returns:
        str: 该活动的详细信息
    """
    try:
        a = await _api_get(f"/activities/{activity_id}")
    except RuntimeError as e:
        return str(e)
    except Exception as e:
        core.logger.exception("Strava 获取活动详情失败")
        return f"[Strava] 获取活动详情失败: {e!s}"

    fields = {
        "名称": a.get("name"),
        "类型": a.get("type"),
        "距离(km)": round((a.get("distance") or 0) / 1000, 2),
        "运动时间(min)": round((a.get("moving_time") or 0) / 60, 1),
        "总时间(min)": round((a.get("elapsed_time") or 0) / 60, 1),
        "爬升(m)": a.get("total_elevation_gain"),
        "平均心率": a.get("average_heartrate"),
        "最大心率": a.get("max_heartrate"),
        "平均速度(km/h)": round((a.get("average_speed") or 0) * 3.6, 2),
        "开始时间": a.get("start_date_local"),
        "描述": a.get("description"),
    }
    lines = [f"{k}: {v}" for k, v in fields.items() if v not in (None, "")]
    return f"[Strava 活动 {activity_id}]\n" + "\n".join(lines)


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="获取Strava运动统计",
    description="获取用户的累计运动统计（今年/历史总里程等）",
)
async def get_athlete_stats(_ctx: AgentCtx) -> str:
    """获取用户的 Strava 累计运动统计

    Returns:
        str: 年度与历史累计统计
    """
    try:
        me = await _api_get("/athlete")
        stats = await _api_get(f"/athletes/{me['id']}/stats")
    except RuntimeError as e:
        return str(e)
    except Exception as e:
        core.logger.exception("Strava 获取统计失败")
        return f"[Strava] 获取统计失败: {e!s}"

    out = [f"[Strava 统计] {me.get('firstname', '')} {me.get('lastname', '')}".strip()]
    for key, label in [
        ("recent_run_totals", "近4周跑步"),
        ("ytd_run_totals", "今年跑步"),
        ("all_run_totals", "历史跑步"),
        ("recent_ride_totals", "近4周骑行"),
        ("ytd_ride_totals", "今年骑行"),
        ("all_ride_totals", "历史骑行"),
    ]:
        t = stats.get(key) or {}
        if t.get("count"):
            out.append(
                f"{label}: {t['count']} 次 | {(t.get('distance') or 0) / 1000:.1f} km | "
                f"{(t.get('moving_time') or 0) / 3600:.1f} h",
            )
    return "\n".join(out)


# --------------------------------------------------------------- webhook


def _pace(seconds: float, metres: float) -> str:
    """Format pace as m:ss /km."""
    km = metres / 1000
    if km < 0.01 or seconds <= 0:
        return "-"
    total = seconds / km
    return f"{int(total // 60)}:{int(total % 60):02d}/km"


def _fmt_splits(a: dict, limit: int) -> str:
    splits = a.get("splits_metric") or []
    if not splits:
        return ""
    lines = []
    for s in splits[:limit]:
        hr = f" | HR {s['average_heartrate']:.0f}" if s.get("average_heartrate") else ""
        elev = s.get("elevation_difference")
        elev_s = f" | {elev:+.0f}m" if elev is not None else ""
        lines.append(
            f"  第{s.get('split')}km: {_pace(s.get('moving_time', 0), s.get('distance', 0))}{hr}{elev_s}",
        )
    if len(splits) > limit:
        lines.append(f"  ...（还有 {len(splits) - limit} 个分段未列出）")
    return "每公里分段:\n" + "\n".join(lines)


async def _fmt_comparison(a: dict, count: int) -> str:
    """Compare this activity against recent ones of the same type."""
    if count <= 0:
        return ""
    try:
        recent = await _api_get("/athlete/activities", {"per_page": count + 5})
    except Exception:
        core.logger.exception("[Strava] 拉取对比数据失败")
        return ""

    same = [
        r
        for r in recent
        if r.get("type") == a.get("type") and r.get("id") != a.get("id") and (r.get("distance") or 0) > 500
    ][:count]
    if not same:
        return ""

    paces = [(r["moving_time"] / (r["distance"] / 1000)) for r in same if r.get("distance") and r.get("moving_time")]
    dists = [r["distance"] / 1000 for r in same if r.get("distance")]
    if not paces:
        return ""

    avg_pace = sum(paces) / len(paces)
    this_pace = (
        a["moving_time"] / (a["distance"] / 1000) if a.get("distance") and a.get("moving_time") else None
    )
    out = [
        f"最近 {len(same)} 次同类型活动参考:",
        f"  平均配速 {int(avg_pace // 60)}:{int(avg_pace % 60):02d}/km | 平均距离 {sum(dists) / len(dists):.2f} km"
        f" | 最长 {max(dists):.2f} km",
    ]
    if this_pace:
        delta = this_pace - avg_pace
        faster = "快" if delta < 0 else "慢"
        out.append(f"  本次配速比平均{faster} {abs(delta):.0f} 秒/km")
    return "\n".join(out)


async def _handle_activity_event(object_id: int, aspect_type: str) -> None:
    """Fetch the activity and push an analysis-ready message into the notify channel.

    Runs detached from the HTTP response -- Strava expects a 200 within ~2s and
    will retry (or disable the subscription) if we block on the API calls.
    """
    chat_key = config.WEBHOOK_NOTIFY_CHAT_KEY
    if not chat_key:
        core.logger.warning("[Strava] 收到活动事件但未配置通知频道，已忽略")
        return

    try:
        a = await _api_get(f"/activities/{object_id}")

        dist = a.get("distance") or 0
        moving = a.get("moving_time") or 0
        core_lines = [
            f"名称: {a.get('name', '')}",
            f"类型: {a.get('type')} | 距离: {dist / 1000:.2f} km | 用时: {moving / 60:.1f} min",
            f"平均配速: {_pace(moving, dist)} | 爬升: {a.get('total_elevation_gain', 0):.0f} m",
            f"开始时间: {a.get('start_date_local', '')}",
        ]
        if a.get("average_heartrate"):
            core_lines.append(
                f"心率: 平均 {a['average_heartrate']:.0f} / 最高 {a.get('max_heartrate', 0):.0f}",
            )
        for key, label in [
            ("average_cadence", "平均步频"),
            ("calories", "消耗(kcal)"),
            ("suffer_score", "强度得分"),
            ("achievement_count", "达成成就数"),
        ]:
            if a.get(key):
                core_lines.append(f"{label}: {a[key]}")

        blocks = ["[Strava 事件] 用户" + ("刚刚完成了一次运动" if aspect_type == "create" else "更新了一次运动")]
        blocks.append("\n".join(core_lines))
        splits = _fmt_splits(a, config.WEBHOOK_MAX_SPLITS)
        if splits:
            blocks.append(splits)
        comparison = await _fmt_comparison(a, config.WEBHOOK_COMPARE_COUNT)
        if comparison:
            blocks.append(comparison)
        blocks.append(config.WEBHOOK_PROMPT)
        text = "\n\n".join(blocks)
    except Exception as e:  # noqa: BLE001 - never let a webhook kill the task
        core.logger.exception("[Strava] 处理活动事件失败")
        text = f"[Strava 事件] 收到活动 {object_id} 的 {aspect_type} 事件，但获取详情失败: {e!s}"

    try:
        await message_service.push_system_message(
            chat_key=chat_key,
            agent_messages=text,
            trigger_agent=True,
        )
        core.logger.info(f"[Strava] 已推送活动 {object_id} 到 {chat_key}")
    except Exception:
        core.logger.exception(f"[Strava] 推送到 {chat_key} 失败")


@plugin.mount_router()
def create_router() -> APIRouter:
    """Strava webhook 路由。

    Strava 订阅校验走 GET 且必须原样回显 hub.challenge，NekroAgent 内置的
    /api/webhook 端点只接受 POST，因此这里单独挂一个插件路由。
    """
    router = APIRouter()

    @router.get("/webhook", summary="Strava 订阅校验")
    async def verify(request: Request):
        params = request.query_params
        mode = params.get("hub.mode")
        token = params.get("hub.verify_token")
        challenge = params.get("hub.challenge")
        core.logger.info(f"[Strava] 收到订阅校验请求 mode={mode}")

        if mode != "subscribe" or token != config.WEBHOOK_VERIFY_TOKEN:
            core.logger.warning("[Strava] 订阅校验失败：mode 或 verify_token 不匹配")
            return JSONResponse({"error": "verification failed"}, status_code=403)

        # must echo verbatim, and the key really is "hub.challenge"
        return JSONResponse({"hub.challenge": challenge})

    @router.post("/webhook", summary="Strava 活动事件")
    async def event(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "bad json"}, status_code=400)

        core.logger.info(f"[Strava] 收到事件: {body}")
        object_type = body.get("object_type")
        aspect_type = body.get("aspect_type")
        object_id = body.get("object_id")

        should_notify = object_type == "activity" and (
            aspect_type == "create" or (aspect_type == "update" and config.WEBHOOK_NOTIFY_ON_UPDATE)
        )
        if should_notify and object_id:
            # respond immediately; do the slow work in the background
            asyncio.create_task(_handle_activity_event(int(object_id), aspect_type))  # noqa: RUF006

        return JSONResponse({"status": "ok"})

    return router


@plugin.mount_cleanup_method()
async def clean_up():
    """清理插件"""
