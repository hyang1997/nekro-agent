"""沙盒退出标记解析

标记由 EXEC_SCRIPT 在 python 进程结束后 echo，永远在末尾。旧实现全文子串搜索 +
按字典序取首个命中，于是脚本自身打印的文本能顶掉真正的标记——两个不同进程的事实
被混成一个。
"""

import pytest

from nekro_agent.models.db_exec_code import ExecStopType
from nekro_agent.services.sandbox.runner import CODE_RUN_END_FLAGS, _split_end_flag

FLAG = CODE_RUN_END_FLAGS


class TestSplitEndFlag:
    @pytest.mark.parametrize("stop_type", list(CODE_RUN_END_FLAGS))
    def test_each_flag_round_trips(self, stop_type: ExecStopType):
        text, parsed = _split_end_flag(f"some output\n{FLAG[stop_type]}")
        assert parsed == stop_type
        assert text == "some output"

    def test_no_flag_returns_none(self):
        text, parsed = _split_end_flag("plain output with no marker")
        assert parsed is None
        assert text == "plain output with no marker"

    def test_empty_output(self):
        text, parsed = _split_end_flag("")
        assert parsed is None
        assert text == ""

    def test_flag_only(self):
        text, parsed = _split_end_flag(FLAG[ExecStopType.NORMAL])
        assert parsed is ExecStopType.NORMAL
        assert text == ""

    def test_trailing_whitespace_tolerated(self):
        _, parsed = _split_end_flag(f"out\n{FLAG[ExecStopType.AGENT]}\n\n  ")
        assert parsed is ExecStopType.AGENT

    def test_flag_appended_without_newline(self):
        """脚本最后一次写入没有换行时，echo 会接在同一行末尾"""
        text, parsed = _split_end_flag(f"no trailing newline{FLAG[ExecStopType.MANUAL]}")
        assert parsed is ExecStopType.MANUAL
        assert text == "no trailing newline"


class TestScriptOutputCannotForgeTheStopType:
    """回归：这正是旧实现判错的那一类输入"""

    def test_printed_normal_flag_does_not_override_real_manual_exit(self):
        output = f"here is a log line: {FLAG[ExecStopType.NORMAL]}\n{FLAG[ExecStopType.MANUAL]}"
        text, parsed = _split_end_flag(output)
        assert parsed is ExecStopType.MANUAL, "脚本打印的标记字面量顶掉了真实退出类型"
        # 真实标记被剥掉；脚本自己打印的那份留在文本里，属于它的输出，不该被改写
        assert not text.endswith(FLAG[ExecStopType.MANUAL])
        assert FLAG[ExecStopType.NORMAL] in text

    @pytest.mark.parametrize("printed", list(CODE_RUN_END_FLAGS))
    @pytest.mark.parametrize("real", list(CODE_RUN_END_FLAGS))
    def test_real_exit_always_wins_over_printed_text(self, printed: ExecStopType, real: ExecStopType):
        _, parsed = _split_end_flag(f"model printed {FLAG[printed]} in its output\n{FLAG[real]}")
        assert parsed == real

    def test_agent_result_is_not_lost_to_a_printed_normal_flag(self):
        """AGENT 轮被误判成 NORMAL 会让插件返回值永远回不到模型"""
        output = f"{FLAG[ExecStopType.NORMAL]} appeared in a file I read\n{FLAG[ExecStopType.AGENT]}"
        _, parsed = _split_end_flag(output)
        assert parsed is ExecStopType.AGENT


class TestFlagSuffixSafety:
    def test_no_flag_is_a_suffix_of_another(self):
        """按长度降序匹配的前提；若将来加了互为后缀的标记，这条会先炸"""
        flags = list(CODE_RUN_END_FLAGS.values())
        for a in flags:
            for b in flags:
                if a is not b:
                    assert not a.endswith(b), f"{a} 以 {b} 结尾，尾部匹配会歧义"

    def test_multimodal_agent_not_confused_with_agent(self):
        _, parsed = _split_end_flag(f"out\n{FLAG[ExecStopType.MULTIMODAL_AGENT]}")
        assert parsed is ExecStopType.MULTIMODAL_AGENT

    def test_normal_is_falsy_and_must_not_be_treated_as_missing(self):
        """ExecStopType.NORMAL == 0，调用方不能用 `parsed or ERROR` 兜底"""
        assert ExecStopType.NORMAL == 0
        assert not bool(ExecStopType.NORMAL)
        _, parsed = _split_end_flag(f"out\n{FLAG[ExecStopType.NORMAL]}")
        assert parsed is not None
        assert parsed is ExecStopType.NORMAL
