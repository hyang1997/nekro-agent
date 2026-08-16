"""
# 健康数据 (Health / Garmin)

给 AI 读取 Hao 的 Garmin 健康数据，用于早报和训练建议。

## 数据来源

**不直接连 Garmin。** 数据来自 `health-dashboard-hao` 仓库里的 `hao.json`——
那是一个 GitHub Actions 每天跑 collector 拉取、经 Zod 校验后提交的文件，已经
包含原始 Garmin API 拿不到的派生指标（ACWR、HRV 基准带、成绩预测历史）。

再接一套 Garmin 认证只会多一个要维护的东西：Garmin 的 refresh token 只活 ~4 天，
仓库那边是每次用账号密码重新登录才绕过这个问题的，没必要复制一遍。

## 时序（重要）

`collect.yml` 在多伦多时间 10:00 才跑，晚于早报。因此仓库另加了一个
`early-pull.yml`，07:30 先拉一次，早报读到的才是**昨晚**的睡眠 / HRV / 身体电量。
本插件会检查 `meta.pullDate` 是否为今天，并在 `data_is_today` 里如实告诉 AI——
数据不新时要说明，不能拿前天的睡眠冒充昨晚的。

## 配置说明

需要一个**细粒度只读** GitHub PAT：仓库限定 `health-dashboard-hao`，权限
`Contents: Read`。填在 `GITHUB_TOKEN` 里。
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
    version="0.1.0",
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
        description="细粒度只读 PAT，仓库限定 health-dashboard-hao，权限 Contents: Read",
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
            "请告诉用户去插件配置里填一个对 health-dashboard-hao 仓库只读的细粒度 PAT。",
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


def _delta(value: Optional[float], base: Optional[float]) -> Optional[float]:
    if value is None or base is None:
        return None
    return round(value - base, 1)


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="今日身体状态",
    description="获取今日睡眠、HRV、静息心率、恢复度、身体电量，含与个人基准的对比",
)
async def get_morning_snapshot(_ctx: AgentCtx) -> Dict[str, Any]:
    """Get this morning's recovery metrics, each compared against Hao's own baseline.

    Every metric comes with its baseline and delta because the raw number alone
    means nothing -- an HRV of 45 ms is only meaningful against his usual band.
    Talk about the comparisons, not the bare figures.

    Check `data_is_today` before speaking. When it is false the file has not been
    refreshed yet and these are an older day's numbers -- say so rather than
    presenting them as last night's.

    Returns:
        dict: {
          "pull_date": "2026-08-15", "data_is_today": true,
          "sleep":   {"hours": 4.79, "score": 60, "baseline_hours": 6.7, "delta_hours": -1.9},
          "hrv":     {"ms": 45, "night_ms": 44, "band_low": 48, "band_high": 61,
                      "vs_band": "below"},
          "resting_hr": {"bpm": 52, "baseline_bpm": 50.7, "delta_bpm": 1.3,
                         "red_flag_bpm": 65},
          "readiness": {"score": 44, "level": "LOW", "recovery_hours": 2.9},
          "body_battery": {"wake": 61, "high": 61, "low": 18},
        }

    Examples:
        snapshot = get_morning_snapshot()
        # -> "You slept 4.8h against your 6.7h norm, HRV is below your 48-61 band,
        #     and resting HR is up 1.3. Today should be easy."
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
        vs_band = None
    elif hrv_ms < band_low:
        vs_band = "below"
    elif hrv_ms > band_high:
        vs_band = "above"
    else:
        vs_band = "within"

    sleep_base = _baseline(data.get("sleep") or [], "hours")
    rhr_base = _baseline(data.get("restingHr") or [], "bpm")

    return {
        "pull_date": pull_date,
        "data_is_today": pull_date == today,
        "sleep": {
            "hours": sleep.get("hours"),
            "score": sleep.get("score"),
            "baseline_hours": sleep_base,
            "delta_hours": _delta(sleep.get("hours"), sleep_base),
        },
        "hrv": {
            "ms": hrv_ms,
            "night_ms": hrv.get("nightMs"),
            "band_low": band_low,
            "band_high": band_high,
            "vs_band": vs_band,
        },
        "resting_hr": {
            "bpm": rhr.get("bpm"),
            "baseline_bpm": rhr_base,
            "delta_bpm": _delta(rhr.get("bpm"), rhr_base),
            "red_flag_bpm": data.get("goal", {}).get("rhrRedFlagBpm"),
        },
        "readiness": {
            "score": readiness.get("readiness"),
            "level": readiness.get("readinessLevel"),
            "recovery_hours": readiness.get("recoveryHours"),
        },
        "body_battery": {
            "wake": readiness.get("bodyBatteryWake"),
            "high": readiness.get("bodyBatteryHigh"),
            "low": readiness.get("bodyBatteryLow"),
        },
    }


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="训练状态",
    description="获取周跑量趋势、急慢性负荷比、训练状态与比赛目标进度",
)
async def get_training_context(_ctx: AgentCtx) -> Dict[str, Any]:
    """Get training load and race-goal context.

    `acwr` (acute:chronic workload ratio) is the injury-risk number: roughly
    0.8-1.3 is the sweet spot, above ~1.5 means load ramped faster than fitness.
    Read it together with `weekly_km` -- a high ratio is only alarming when the
    week-over-week jump explains it.

    Returns:
        dict: {
          "phase": "BASE PHASE",
          "race": {"name": ..., "date": "2026-10-18", "days_out": 64, "goal": "SUB-2:00 HALF"},
          "weekly_km": [{"week_start": ..., "km": 27.9}, ...],
          "week_over_week_pct": 67.1,
          "acwr": 1.5, "acwr_status": "HIGH",
          "training_status": "PRODUCTIVE_10", "load_focus": "ANAEROBIC_SHORTAGE",
          "predictions": {"half": "1:50:26", "half_delta_min": -10, ...},
        }
    """
    data = await _fetch_profile()

    weekly = data.get("weeklyMileage") or []
    recent = [{"week_start": w.get("weekStart"), "km": w.get("km")} for w in weekly[-4:]]

    wow = None
    if len(weekly) >= 2 and weekly[-2].get("km"):
        wow = round((weekly[-1]["km"] - weekly[-2]["km"]) / weekly[-2]["km"] * 100, 1)

    # events 里挑当前目标赛事，用于算距离比赛还有多少天
    race: Dict[str, Any] = {}
    active_id = data.get("activeEventId")
    for event in data.get("events") or []:
        if event.get("id") == active_id:
            race = {"name": event.get("name"), "date": event.get("raceDate"), "goal": event.get("goalLabel")}
            if event.get("raceDate"):
                try:
                    race_date = datetime.date.fromisoformat(event["raceDate"])
                    race["days_out"] = (race_date - datetime.date.today()).days
                except ValueError:
                    core.logger.warning(f"[Health] 无法解析比赛日期: {event.get('raceDate')}")
            break

    rp = data.get("racePredictions") or {}
    return {
        "phase": data.get("meta", {}).get("phase"),
        "race": race,
        "weekly_km": recent,
        "week_over_week_pct": wow,
        "acwr": rp.get("acuteChronicRatio"),
        "acwr_status": rp.get("acwrStatus"),
        "training_status": rp.get("trainingStatus"),
        "load_focus": (rp.get("loadFocus") or {}).get("balance"),
        "predictions": {
            "five_k": rp.get("fiveK"),
            "ten_k": rp.get("tenK"),
            "half": rp.get("half"),
            "half_delta_min": rp.get("halfDeltaMin"),
            "marathon": rp.get("marathon"),
        },
    }


@plugin.mount_cleanup_method()
async def clean_up():
    """清理插件"""
    _cache["at"] = 0.0
    _cache["data"] = None
