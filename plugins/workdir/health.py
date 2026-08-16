"""
# 健康数据 (Health / Garmin)

给 AI 读取 Hao 的 Garmin 健康数据，用于早报和训练建议。

## 数据来源

**不直接连 Garmin。** 数据来自 `health-dashboard-hao` 仓库里的 `hao.json`——
那是一个 GitHub Actions 跑 collector 拉取、经 Zod 校验后提交的文件，已经包含
原始 Garmin API 拿不到的派生指标（ACWR、HRV 基准带、成绩预测历史）。

再接一套 Garmin 认证只会多一个要维护的东西：Garmin 的 refresh token 只活 ~4 天，
仓库那边是每次用账号密码重新登录才绕过这个问题的，没必要复制一遍。

## 时序

`collect.yml` 在多伦多时间 10:00 才跑，晚于早报；而 GitHub 的定时器实测会迟到
25-60 分钟，靠 cron 抢在早报前面并不可靠。因此改由 Agent 自己在早报前几分钟调用
`refresh_health_data()` 触发 `agent-refresh.yml`（实测约 25 秒即完成数据提交）。

本插件会检查 `meta.pullDate` 是否为今天，并在输出里如实标注——数据不新时要说明，
不能拿前天的睡眠冒充昨晚的。

## 方法类型（重要）

`get_health_data` 必须是 **AGENT** 而不是 TOOL。TOOL 的返回值只回到沙盒脚本里，
意味着模型得在**看到数据之前**就把点评写好；AGENT 会把结果送回上下文并触发下一轮，
模型才能真正基于数据说话。第一版用了 TOOL，结果就是数据取到了、一个字也没发出来。

同理它只有一个方法而不是拆成两个：AGENT 方法会中断脚本（其后的代码不执行），
拆成两个就要多烧一轮迭代，而 `AI_SCRIPT_MAX_RETRY_TIMES` 默认只有 3。

## 配置说明

需要一个细粒度 GitHub PAT，仓库限定 `health-dashboard-hao`，权限
`Contents: Read`（读数据）+ `Actions: Write`（触发刷新）。填在 `GITHUB_TOKEN` 里。

`Actions: Write` 是有代价的：Agent 的输入里有 Discord 消息这类不可信内容，被注入时
最坏情况是反复触发工作流、烧 Actions 额度。工作流侧用 `concurrency` 兜底，不会并发
跑出竞态。仓库 secrets（Garmin 账密）不会因此泄露——它们不进日志。
只想要只读时把权限降到 `Contents: Read`，`refresh_health_data` 会明确报错。
"""

import datetime
import statistics
import time
from typing import Any, Dict, List, Optional

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

plugin = NekroPlugin(
    name="健康数据",
    module_name="health",
    description="读取 Garmin 健康与训练数据（睡眠 / HRV / 恢复 / 跑量 / 目标）",
    version="0.2.0",
    author="Hao",
    url="https://github.com/hyang1997/health-dashboard-hao",
    i18n_name=i18n.i18n_text(zh_CN="健康数据", en_US="Health Data"),
    i18n_description=i18n.i18n_text(
        zh_CN="读取 Garmin 健康与训练数据",
        en_US="Read Garmin health and training data",
    ),
    allow_sleep=True,
    sleep_brief="提供睡眠 / HRV / 恢复度 / 跑量 / 比赛目标数据。早报或聊到身体状态、训练时激活。",
)


@plugin.mount_config()
class HealthConfig(ConfigBase):
    """健康数据配置"""

    GITHUB_TOKEN: str = Field(
        default="",
        title="GitHub 访问令牌",
        description="细粒度 PAT，仓库限定 health-dashboard-hao，权限 Contents: Read + Actions: Write",
        json_schema_extra=ExtraField(
            is_secret=True,
            required=True,
            i18n_title=i18n.i18n_text(zh_CN="GitHub 访问令牌", en_US="GitHub Token"),
        ).model_dump(),
    )
    REPO: str = Field(
        default="hyang1997/health-dashboard-hao",
        title="数据仓库",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="数据仓库", en_US="Data Repository"),
        ).model_dump(),
    )
    PROFILE_PATH: str = Field(
        default="src/data/profiles/hao.json",
        title="数据文件路径",
        description="仓库内的 profile JSON 路径。换成 jessie.json 即可读另一位。",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="数据文件路径", en_US="Profile Path"),
        ).model_dump(),
    )
    WORKFLOW_FILE: str = Field(
        default="agent-refresh.yml",
        title="刷新用的工作流",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="刷新用的工作流", en_US="Refresh Workflow"),
        ).model_dump(),
    )
    WORKFLOW_REF: str = Field(
        default="master",
        title="工作流分支",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="工作流分支", en_US="Workflow Ref"),
        ).model_dump(),
    )
    BASELINE_DAYS: int = Field(
        default=7,
        title="基准天数",
        description="计算睡眠 / HRV / 静息心率基准所用的天数（不含当天）",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="基准天数", en_US="Baseline Days"),
        ).model_dump(),
    )
    CACHE_SECONDS: int = Field(
        default=900,
        title="缓存时间",
        description="单位: 秒。同一轮对话里多次调用不必重复拉取。",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="缓存时间", en_US="Cache Seconds"),
        ).model_dump(),
    )


config: HealthConfig = plugin.get_config(HealthConfig)

_cache: Dict[str, Any] = {"at": 0.0, "data": None}


async def _fetch_profile() -> Dict[str, Any]:
    """从 GitHub 取 profile JSON。

    读的是**远端**而不是本地 clone——本地工作副本随时可能落后好几天，
    静默地拿旧数据当今天的用，比报错难发现得多。
    """
    if not config.GITHUB_TOKEN:
        raise RuntimeError(
            "[Health] 未配置 GITHUB_TOKEN。这不是临时错误，重试无用——"
            "请告诉用户去插件配置里填一个对 health-dashboard-hao 仓库的细粒度 PAT。",
        )

    if _cache["data"] is not None and time.time() - _cache["at"] < config.CACHE_SECONDS:
        return _cache["data"]

    url = f"https://api.github.com/repos/{config.REPO}/contents/{config.PROFILE_PATH}"
    async with AsyncClient() as client:
        response = await client.get(
            url,
            headers={
                "Authorization": f"Bearer {config.GITHUB_TOKEN}",
                # raw media type 直接返回文件内容，省掉 base64，也避开 contents API
                # 对 base64 响应的 1MB 限制
                "Accept": "application/vnd.github.raw",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=Timeout(read=30, write=30, connect=10, pool=10),
        )

    if response.status_code == 401:
        raise RuntimeError("[Health] GitHub token 无效或已过期。这不是临时错误，请让用户重新生成 PAT。")
    if response.status_code == 404:
        raise RuntimeError(
            f"[Health] 找不到 {config.REPO}/{config.PROFILE_PATH}。"
            "这不是临时错误——可能是路径变了，或 PAT 没有授权到这个仓库。",
        )
    response.raise_for_status()

    data = response.json()
    _cache["at"] = time.time()
    _cache["data"] = data
    return data


def _baseline(series: List[Dict[str, Any]], key: str) -> Optional[float]:
    """取最近 N 天（不含当天）的均值作为基准。"""
    window = [row[key] for row in series[-(config.BASELINE_DAYS + 1) : -1] if row.get(key) is not None]
    return round(statistics.mean(window), 1) if window else None


def _signed(value: Optional[float]) -> str:
    return "n/a" if value is None else (f"+{value}" if value > 0 else str(value))


def _delta(value: Optional[float], base: Optional[float]) -> Optional[float]:
    if value is None or base is None:
        return None
    return round(value - base, 1)


@plugin.mount_sandbox_method(
    SandboxMethodType.BEHAVIOR,
    name="刷新健康数据",
    description="触发一次 Garmin 拉取，让数据更新到最新",
)
async def refresh_health_data(_ctx: AgentCtx) -> str:
    """Start a fresh Garmin pull. Returns immediately -- it does NOT wait.

    The pull commits in roughly 30 seconds. Do not call this and then read the
    data in the same breath; you will get the previous pull. The morning job
    calls this a few minutes before the brief.

    Only worth calling once a day, before the morning brief. Sleep, HRV,
    readiness and body battery are computed once after waking -- refreshing
    again later in the day returns the same numbers.

    Returns:
        str: Confirmation that the refresh was queued.
    """
    if not config.GITHUB_TOKEN:
        raise RuntimeError(
            "[Health] 未配置 GITHUB_TOKEN，无法触发刷新。这不是临时错误，重试无用——请告知用户。",
        )

    url = f"https://api.github.com/repos/{config.REPO}/actions/workflows/{config.WORKFLOW_FILE}/dispatches"
    async with AsyncClient() as client:
        response = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {config.GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={"ref": config.WORKFLOW_REF},
            timeout=Timeout(read=30, write=30, connect=10, pool=10),
        )

    if response.status_code == 403:
        raise RuntimeError(
            "[Health] token 缺少 Actions: Write 权限，无法触发工作流。"
            "这不是临时错误，重试无用——请让用户给 PAT 加上该权限。",
        )
    if response.status_code == 404:
        raise RuntimeError(
            f"[Health] 找不到工作流 {config.WORKFLOW_FILE}，或 token 无权访问。这不是临时错误。",
        )
    response.raise_for_status()

    # 拉取完成后仓库内容会变，本地缓存必须作废，否则下一次读到的还是旧的
    _cache["at"] = 0.0
    _cache["data"] = None

    core.logger.info(f"[Health] 已触发 {config.WORKFLOW_FILE} 刷新")
    return "Garmin refresh queued; fresh data commits in about 30 seconds."


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="读取健康数据",
    description="读取今日睡眠、HRV、静息心率、恢复度与训练负荷，含与个人基准的对比",
)
async def get_health_data(_ctx: AgentCtx) -> str:
    """Read today's recovery and training numbers, each against Hao's own baseline.

    Returns a text block, which lands back in your context so you can write
    about it on the next step. Every figure is paired with its baseline or
    normal band because the bare number means nothing on its own -- an HRV of
    45 ms is only low relative to a 48-61 band. Talk about the comparisons and
    what they imply for today's training; do not read the numbers back.

    If the block says the data is NOT from today, say so rather than passing an
    older day's sleep off as last night's.

    Returns:
        str: Formatted recovery + training summary.
    """
    data = await _fetch_profile()

    pull_date = data.get("meta", {}).get("pullDate", "")
    today = datetime.date.today().isoformat()

    sleep = (data.get("sleep") or [{}])[-1]
    hrv = (data.get("hrv") or [{}])[-1]
    rhr = (data.get("restingHr") or [{}])[-1]
    readiness = (data.get("readiness") or [{}])[-1]

    band_low, band_high = hrv.get("bandLowMs"), hrv.get("bandHighMs")
    hrv_ms = hrv.get("ms")
    if hrv_ms is None or band_low is None or band_high is None:
        vs_band = "unknown"
    elif hrv_ms < band_low:
        vs_band = "BELOW his normal band"
    elif hrv_ms > band_high:
        vs_band = "above his normal band"
    else:
        vs_band = "within his normal band"

    sleep_base = _baseline(data.get("sleep") or [], "hours")
    rhr_base = _baseline(data.get("restingHr") or [], "bpm")

    weekly = data.get("weeklyMileage") or []
    wow = None
    if len(weekly) >= 2 and weekly[-2].get("km"):
        wow = round((weekly[-1]["km"] - weekly[-2]["km"]) / weekly[-2]["km"] * 100, 1)

    race_line = ""
    for event in data.get("events") or []:
        if event.get("id") == data.get("activeEventId"):
            days_out = ""
            if event.get("raceDate"):
                try:
                    delta_days = (datetime.date.fromisoformat(event["raceDate"]) - datetime.date.today()).days
                    days_out = f", {delta_days} days out"
                except ValueError:
                    core.logger.warning(f"[Health] 无法解析比赛日期: {event.get('raceDate')}")
            race_line = f"{event.get('name')} on {event.get('raceDate')}{days_out}, goal {event.get('goalLabel')}"
            break

    rp = data.get("racePredictions") or {}

    # pullDate 是 collector 按 UTC 打的，本地是多伦多时间，所以晚上 20:00 之后
    # UTC 已经是第二天——用 == 判断会把刚拉的新数据判成过期。只有严格早于本地
    # 今天才算旧。
    try:
        days_stale = (datetime.date.fromisoformat(today) - datetime.date.fromisoformat(pull_date)).days
    except ValueError:
        days_stale = None

    if days_stale is None:
        freshness = f"Data pull date unreadable ({pull_date!r}); treat freshness as unknown."
    elif days_stale <= 0:
        freshness = f"Data pulled {pull_date} -- current, these are last night's numbers."
    else:
        freshness = (
            f"WARNING: data is {days_stale} day(s) stale (pulled {pull_date}, today is {today}). "
            "These are NOT last night's numbers -- say so instead of presenting them as this morning's."
        )

    return "\n".join(
        [
            freshness,
            "",
            "RECOVERY",
            f"  sleep       {sleep.get('hours')} h (score {sleep.get('score')}) | "
            f"{config.BASELINE_DAYS}-day baseline {sleep_base} h | {_signed(_delta(sleep.get('hours'), sleep_base))} h",
            f"  HRV         {hrv_ms} ms (overnight {hrv.get('nightMs')}) | "
            f"normal band {band_low}-{band_high} ms | {vs_band}",
            f"  resting HR  {rhr.get('bpm')} bpm | baseline {rhr_base} bpm | "
            f"{_signed(_delta(rhr.get('bpm'), rhr_base))} bpm | red flag at {data.get('goal', {}).get('rhrRedFlagBpm')}",
            f"  readiness   {readiness.get('readiness')} ({readiness.get('readinessLevel')}), "
            f"needs {readiness.get('recoveryHours')} h more recovery",
            f"  body batt.  woke at {readiness.get('bodyBatteryWake')}, "
            f"overnight low {readiness.get('bodyBatteryLow')}, peak {readiness.get('bodyBatteryHigh')}",
            "",
            "TRAINING",
            f"  phase       {data.get('meta', {}).get('phase')}",
            f"  race        {race_line}",
            "  weekly km   " + " -> ".join(f"{w.get('km')}" for w in weekly[-4:])
            + (f"  (latest week {_signed(wow)}% vs previous)" if wow is not None else ""),
            f"  ACWR        {rp.get('acuteChronicRatio')} ({rp.get('acwrStatus')}) "
            "- acute:chronic load; 0.8-1.3 is the safe zone, above ~1.5 means load outran fitness",
            f"  status      {rp.get('trainingStatus')}, load focus {(rp.get('loadFocus') or {}).get('balance')}",
            f"  half pred   {rp.get('half')} ({_signed(rp.get('halfDeltaMin'))} min vs goal), "
            f"5K {rp.get('fiveK')}, 10K {rp.get('tenK')}",
        ],
    )


@plugin.mount_cleanup_method()
async def clean_up():
    """清理插件"""
    _cache["at"] = 0.0
    _cache["data"] = None
