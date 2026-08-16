"""沙盒运行时会话登记表

RPC 网关靠它回答"这次调用来自哪个容器、代表哪个频道"。

原先 `/ext/rpc_exec` 把 `container_key` 和 `from_chat_key` 当作查询参数直接采信，而唯一的
凭证 RPC_SECRET_KEY 是全局共享的、且以裸全局变量的形式出现在模型脚本的命名空间里。
于是一个频道的沙盒代码只要换个 `from_chat_key` 就能以另一个频道的身份调用插件方法——
令牌只证明了"你是某个沙盒"，没有证明"你是哪个沙盒"。

这里给每次容器运行发一枚一次性令牌，服务端保存 令牌 -> (容器, 频道) 的绑定，
RPC 网关只认这份绑定，不再采信调用方自报的身份。
"""

from typing import Dict, NamedTuple, Optional


class SandboxSession(NamedTuple):
    """一次沙盒运行的权威身份"""

    container_key: str
    chat_key: str


# 令牌 -> 会话。仅在容器运行期间存在，进程内存即可，无需持久化。
_sessions: Dict[str, SandboxSession] = {}


def register(token: str, container_key: str, chat_key: str) -> None:
    """在容器启动前登记本次运行的身份"""
    _sessions[token] = SandboxSession(container_key=container_key, chat_key=chat_key)


def resolve(token: str) -> Optional[SandboxSession]:
    """按令牌取回权威身份，未登记返回 None"""
    if not token:
        return None
    return _sessions.get(token)


def unregister(token: str) -> None:
    """容器结束后注销令牌，令牌不可复用"""
    _sessions.pop(token, None)


def active_count() -> int:
    """当前有效令牌数，用于诊断泄漏"""
    return len(_sessions)
