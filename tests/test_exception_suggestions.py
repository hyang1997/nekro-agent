"""失败回传给模型时附带的恢复建议

原先 exception_suggestion_map 只有 SyntaxError 一条。这里覆盖补齐后的映射，
重点是"建议必须是可执行的下一步"，以及匹配顺序不能被父类异常抢走。
"""

import pytest

from nekro_agent.models.db_exec_code import ExecStopType
from nekro_agent.services.agent.run_agent import (
    _EXCEPTION_SUGGESTIONS,
    _STOP_TYPE_SUGGESTIONS,
    _resolve_suggestion,
)

TB = "Traceback (most recent call last):\n  File \"run_script.py\", line 3, in <module>\n"


class TestResolveSuggestion:
    def test_no_match_returns_empty(self):
        assert _resolve_suggestion(f"{TB}ZeroDivisionError: division by zero", ExecStopType.ERROR, 120) == ""

    def test_syntax_error(self):
        out = _resolve_suggestion(f"{TB}SyntaxError: invalid syntax", ExecStopType.ERROR, 120)
        assert "syntax" in out.lower()

    def test_indentation_error_wins_over_syntax_error(self):
        """IndentationError 是 SyntaxError 的子类，必须先命中更具体的那条"""
        out = _resolve_suggestion(f"{TB}IndentationError: unexpected indent", ExecStopType.ERROR, 120)
        assert "indentation" in out.lower()
        assert out != dict(_EXCEPTION_SUGGESTIONS)["SyntaxError"]

    def test_module_not_found_points_at_dynamic_importer(self):
        out = _resolve_suggestion(f"{TB}ModuleNotFoundError: No module named 'bs4'", ExecStopType.ERROR, 120)
        assert "dynamic_importer" in out

    def test_name_error_forbids_inventing_methods(self):
        out = _resolve_suggestion(f"{TB}NameError: name 'fetch_weather' is not defined", ExecStopType.ERROR, 120)
        assert "invent" in out.lower()

    def test_file_not_found_gives_a_runnable_probe(self):
        out = _resolve_suggestion(f"{TB}FileNotFoundError: [Errno 2] './shared/x.txt'", ExecStopType.ERROR, 120)
        assert "os.listdir" in out
        assert "exit(9)" in out

    def test_type_error_points_at_the_signature(self):
        out = _resolve_suggestion(f"{TB}TypeError: send_msg_text() takes 2 args", ExecStopType.ERROR, 120)
        assert "signature" in out.lower()


class TestTimeoutSuggestion:
    def test_timeout_uses_stop_type_not_traceback(self):
        """超时的输出里没有异常名，建议必须由退出类型给出"""
        out = _resolve_suggestion("# container killed", ExecStopType.TIMEOUT, 120)
        assert "120 seconds" in out

    def test_timeout_carries_the_configured_value(self):
        assert "45 seconds" in _resolve_suggestion("", ExecStopType.TIMEOUT, 45)

    def test_stop_type_beats_a_traceback_match(self):
        """超时的脚本里可能恰好打印过 'SyntaxError' 字样，超时建议优先"""
        out = _resolve_suggestion("checking for SyntaxError...", ExecStopType.TIMEOUT, 120)
        assert "sandbox is killed" in out.lower()


class TestSuggestionContent:
    @pytest.mark.parametrize(("marker", "suggestion"), _EXCEPTION_SUGGESTIONS)
    def test_every_suggestion_is_non_trivial(self, marker: str, suggestion: str):
        assert len(suggestion) > 40, f"{marker} 的建议太短，说不清下一步"

    @pytest.mark.parametrize(("marker", "suggestion"), _EXCEPTION_SUGGESTIONS)
    def test_suggestions_do_not_leak_format_placeholders(self, marker: str, suggestion: str):
        """只有 _STOP_TYPE_SUGGESTIONS 会走 .format()，异常建议里的花括号会原样发给模型"""
        assert "{" not in suggestion, f"{marker} 的建议里有未替换的占位符"

    def test_timeout_template_only_uses_known_placeholder(self):
        assert _STOP_TYPE_SUGGESTIONS[ExecStopType.TIMEOUT].format(timeout=1).count("{") == 0

    def test_more_specific_markers_come_first(self):
        markers = [m for m, _ in _EXCEPTION_SUGGESTIONS]
        assert markers.index("IndentationError") < markers.index("SyntaxError")
        assert markers.index("ModuleNotFoundError") < markers.index("ImportError")
