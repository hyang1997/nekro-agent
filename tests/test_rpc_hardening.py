"""RPC 网关加固

两个缺陷都源于"令牌证明得太少"：
1. 请求体裸 pickle.loads，且发生在 schema 校验之前 —— 沙盒里跑的是模型生成的代码，
   而 nekro_agent 容器挂着可读写的 docker.sock。
2. 调用方自报 container_key / from_chat_key，服务端直接采信 —— 一个频道的沙盒
   可以冒充另一个频道调用插件方法。
"""

import io
import pickle

import pytest

from nekro_agent.schemas.errors import ValidationError
from nekro_agent.services.rpc_service import (
    RestrictedUnpickler,
    decode_rpc_request,
    safe_pickle_loads,
)
from nekro_agent.services.sandbox import session_registry


def _write_marker(path: str) -> None:
    """对照组 payload 的落点。模块级函数才能被 pickle 引用，也正好走 find_class"""
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("executed")


class _MarkerGadget:
    """反序列化时写一个标记文件，用来观测 payload 是否真的被执行"""

    def __init__(self, path: str) -> None:
        self.path = path

    def __reduce__(self):
        return (_write_marker, (self.path,))


class _Detonator:
    """模拟 pickle gadget：反序列化时会调用 os.system。

    这里用 echo 作为无害载荷；真正的重点是它根本不该被执行到。
    """

    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


class TestRestrictedUnpickler:
    def test_plain_data_still_loads(self):
        payload = {"method": "send_msg_text", "args": ["chat", "hi"], "kwargs": {"n": 1}}
        assert safe_pickle_loads(pickle.dumps(payload)) == payload

    @pytest.mark.parametrize(
        "value",
        [
            None,
            True,
            42,
            3.14,
            "文本",
            b"bytes",
            [1, "a", None],
            (1, 2),
            {"k": [1, {"n": None}]},
            set(),
        ],
    )
    def test_pure_data_types_round_trip(self, value):
        assert safe_pickle_loads(pickle.dumps(value)) == value

    def test_reduce_gadget_is_blocked(self):
        with pytest.raises(pickle.UnpicklingError):
            safe_pickle_loads(pickle.dumps(_Detonator()))

    def test_arbitrary_global_is_blocked(self):
        with pytest.raises(pickle.UnpicklingError):
            safe_pickle_loads(pickle.dumps(pickle.dumps))

    def test_find_class_always_raises(self):
        u = RestrictedUnpickler(io.BytesIO(b""))
        with pytest.raises(pickle.UnpicklingError):
            u.find_class("os", "system")
        with pytest.raises(pickle.UnpicklingError):
            u.find_class("builtins", "print")


class TestGuardIsNotVacuous:
    """对照组：证明旧路径确实会执行 payload，新路径确实不会。

    没有这一组，上面的 raises 断言可能只是因为 payload 本来就构造失败。
    payload 用写临时文件代替真实危害。
    """

    @staticmethod
    def _marker_payload(marker):
        return pickle.dumps(_MarkerGadget(marker))

    def test_plain_pickle_loads_executes_the_payload(self, tmp_path):
        marker = tmp_path / "old_path.txt"
        pickle.loads(self._marker_payload(str(marker)))  # noqa: S301
        assert marker.exists(), "对照组失效：payload 没有真的执行，下面的断言就没有意义"

    def test_restricted_unpickler_does_not_execute_the_payload(self, tmp_path):
        marker = tmp_path / "new_path.txt"
        with pytest.raises(pickle.UnpicklingError):
            safe_pickle_loads(self._marker_payload(str(marker)))
        assert not marker.exists(), "payload 仍然被执行了"

    def test_decode_rpc_request_does_not_execute_the_payload(self, tmp_path):
        marker = tmp_path / "gateway.txt"
        with pytest.raises(ValidationError):
            decode_rpc_request(self._marker_payload(str(marker)))
        assert not marker.exists(), "网关入口仍然会执行 payload"


class TestDecodeRpcRequest:
    def test_valid_request(self):
        raw = pickle.dumps({"method": "m", "args": [1], "kwargs": {"a": 2}})
        req = decode_rpc_request(raw)
        assert req.method == "m"
        assert req.args == [1]
        assert req.kwargs == {"a": 2}

    def test_gadget_becomes_validation_error_not_execution(self):
        """gadget 必须在解包阶段就被挡下，而不是先执行再校验"""
        with pytest.raises(ValidationError):
            decode_rpc_request(pickle.dumps(_Detonator()))

    def test_garbage_body(self):
        with pytest.raises(ValidationError):
            decode_rpc_request(b"not a pickle at all")

    def test_empty_body(self):
        with pytest.raises(ValidationError):
            decode_rpc_request(b"")

    def test_wrong_shape_is_rejected(self):
        with pytest.raises(ValidationError):
            decode_rpc_request(pickle.dumps({"nope": 1}))

    def test_validation_happens_after_safe_decode(self):
        """纯数据但结构不对 -> 走 pydantic 校验失败，而不是解包失败"""
        with pytest.raises(ValidationError):
            decode_rpc_request(pickle.dumps(["method", "args"]))


class TestSessionRegistry:
    def setup_method(self):
        for token in list(session_registry._sessions):  # noqa: SLF001
            session_registry.unregister(token)

    def test_register_resolve_unregister(self):
        session_registry.register("tok", "sandbox_abc", "discord-123")
        session = session_registry.resolve("tok")
        assert session is not None
        assert session.container_key == "sandbox_abc"
        assert session.chat_key == "discord-123"

        session_registry.unregister("tok")
        assert session_registry.resolve("tok") is None

    def test_unknown_token_resolves_to_none(self):
        assert session_registry.resolve("never-issued") is None

    def test_empty_token_resolves_to_none(self):
        """缺失 header 会变成空串，不能因此匹配到任何会话"""
        assert session_registry.resolve("") is None

    def test_token_is_not_reusable_after_run(self):
        session_registry.register("tok", "sandbox_abc", "discord-123")
        session_registry.unregister("tok")
        assert session_registry.resolve("tok") is None

    def test_unregister_is_idempotent(self):
        session_registry.unregister("never-issued")
        session_registry.register("tok", "c", "k")
        session_registry.unregister("tok")
        session_registry.unregister("tok")
        assert session_registry.active_count() == 0

    def test_sessions_are_isolated(self):
        session_registry.register("tok_a", "sandbox_a", "discord-a")
        session_registry.register("tok_b", "sandbox_b", "discord-b")
        assert session_registry.resolve("tok_a").chat_key == "discord-a"  # type: ignore[union-attr]
        assert session_registry.resolve("tok_b").chat_key == "discord-b"  # type: ignore[union-attr]

    def test_one_channel_token_cannot_name_another_channel(self):
        """核心不变式：身份来自登记表，调用方自报的 chat_key 无从影响它"""
        session_registry.register("tok_a", "sandbox_a", "discord-a")
        resolved = session_registry.resolve("tok_a")
        assert resolved is not None
        # 无论调用方在查询串里写什么，可用的只有这一个 chat_key
        assert resolved.chat_key == "discord-a"


class TestSandboxTokenPlumbing:
    def test_generated_api_caller_carries_a_token_placeholder(self):
        """模板必须留有占位符，否则沙盒发不出 X-Container-Token"""
        from pathlib import Path

        src = Path("nekro_agent/services/sandbox/ext_caller_code.py").read_text(encoding="utf-8")
        assert '{CONTAINER_TOKEN}' in src
        assert '"X-Container-Token": CONTAINER_TOKEN' in src
