"""代码围栏剥离的回归测试。

背景：提取正则要求围栏成对闭合，模型只开不闭时整段响应会连同 ``` 行被当作
代码执行，抛 SyntaxError；模型看不到自己多写了围栏，只会原样重试，表现为
一次无声的死循环。
"""

import pytest

from nekro_agent.services.agent.resolver import parse_chat_response

FENCE = "```"


def _code(raw: str) -> str:
    return parse_chat_response(raw).code_content


def test_closed_fence_still_works():
    """闭合围栏是既有主路径，不能被回归破坏。"""
    raw = f"{FENCE}python\nsend_msg_text(_ck, 'hi')\n{FENCE}"
    assert _code(raw) == "send_msg_text(_ck, 'hi')"


def test_plain_code_untouched():
    """不含围栏的响应必须原样通过。"""
    raw = "send_msg_text(_ck, 'hi')"
    assert _code(raw) == "send_msg_text(_ck, 'hi')"


@pytest.mark.parametrize(
    "opening",
    [f"{FENCE}python", f"{FENCE}py", f"{FENCE}Python", FENCE],
    ids=["python", "py", "Python", "bare"],
)
def test_unterminated_fence_is_stripped(opening: str):
    """只开不闭：各种语言标注都要剥掉，且不能残留围栏。"""
    code = _code(f"{opening}\nsend_msg_text(_ck, 'hi')")
    assert code == "send_msg_text(_ck, 'hi')"
    assert FENCE not in code


def test_preamble_before_unterminated_fence_is_dropped():
    """围栏前的自然语言是说给人听的，不是代码——旧实现在这里失败。"""
    code = _code(f"好的，代码如下：\n{FENCE}python\nsend_msg_text(_ck, 'hi')")
    assert code == "send_msg_text(_ck, 'hi')"
    assert "好的" not in code


def test_stray_closing_fence_is_stripped():
    """多出一条收尾围栏也不应进入执行内容。

    与 test_preamble_before_unterminated_fence_is_dropped 构成一对：同样是
    "第一条围栏"，这里它是落单的收尾符，前面的内容是代码，不能丢。
    """
    code = _code(f"send_msg_text(_ck, 'hi')\n{FENCE}")
    assert code == "send_msg_text(_ck, 'hi')"


def test_preamble_before_bare_fence_is_dropped():
    """裸围栏后面还有内容，说明它是开围栏，前面的自然语言要丢。"""
    code = _code(f"好的，代码如下：\n{FENCE}\nsend_msg_text(_ck, 'hi')")
    assert code == "send_msg_text(_ck, 'hi')"


def test_fence_with_think_tags():
    """思维链 + 只开不闭的围栏。"""
    raw = f"<think>先打个招呼</think>\n{FENCE}python\nsend_msg_text(_ck, 'hi')"
    parsed = parse_chat_response(raw)
    assert parsed.code_content == "send_msg_text(_ck, 'hi')"
    assert parsed.thought_chain == "先打个招呼"


def test_multiline_body_preserved():
    """剥离围栏不能破坏缩进或行序。"""
    body = "for i in range(3):\n    send_msg_text(_ck, str(i))"
    assert _code(f"{FENCE}python\n{body}") == body
