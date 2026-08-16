"""连续重复提交同一份脚本的提醒

一轮迭代 = 一次完整生成 + 一次容器启动。模型把上一轮的报错脚本原样再跑一遍，
基本等同于没读输出，而框架此前对此毫无反应，只会安静地烧完全部迭代次数。
"""

import pytest

from nekro_agent.services.agent.run_agent import (
    _REPEAT_THRESHOLDS,
    _canonical_code,
    _repeat_reminder,
)


class TestCanonicalCode:
    def test_identical_code_matches(self):
        code = "print('hi')\nexit(9)"
        assert _canonical_code(code) == _canonical_code(code)

    def test_trailing_whitespace_ignored(self):
        assert _canonical_code("print('hi')   \n") == _canonical_code("print('hi')\n")

    def test_blank_lines_ignored(self):
        assert _canonical_code("a=1\n\n\nb=2") == _canonical_code("a=1\nb=2")

    def test_crlf_matches_lf(self):
        assert _canonical_code("a=1\r\nb=2\r\n") == _canonical_code("a=1\nb=2\n")

    def test_indentation_is_significant(self):
        """缩进在 Python 里是语义，不能归一化掉"""
        assert _canonical_code("if x:\n    a=1") != _canonical_code("if x:\na=1")

    def test_different_code_differs(self):
        assert _canonical_code("print(1)") != _canonical_code("print(2)")

    def test_empty_code_is_empty(self):
        assert _canonical_code("") == ""
        assert _canonical_code("\n\n  \n") == ""


class TestRepeatReminder:
    def test_first_submission_is_silent(self):
        assert _repeat_reminder(1) == ""

    def test_zero_is_silent(self):
        assert _repeat_reminder(0) == ""

    def test_first_threshold_gives_short_nudge(self):
        out = _repeat_reminder(_REPEAT_THRESHOLDS[0])
        assert out
        assert "same script" in out
        assert "{count}" not in out

    def test_later_threshold_names_the_count(self):
        out = _repeat_reminder(_REPEAT_THRESHOLDS[1])
        assert f"{_REPEAT_THRESHOLDS[1]} times in a row" in out

    def test_between_thresholds_is_silent(self):
        between = _REPEAT_THRESHOLDS[0] + 1
        assert between not in _REPEAT_THRESHOLDS
        assert _repeat_reminder(between) == ""

    def test_past_highest_threshold_keeps_reminding(self):
        """迭代上限只有个位数，越界后剩下的几轮正是最该被打断的"""
        out = _repeat_reminder(_REPEAT_THRESHOLDS[-1] + 3)
        assert out
        assert f"{_REPEAT_THRESHOLDS[-1] + 3} times in a row" in out

    def test_escalates_from_generic_to_detailed(self):
        assert _repeat_reminder(_REPEAT_THRESHOLDS[0]) != _repeat_reminder(_REPEAT_THRESHOLDS[1])

    def test_reminder_offers_an_exit(self):
        """提醒必须给出"换路子或者收口"两条路，不能只是骂一句"""
        out = _repeat_reminder(_REPEAT_THRESHOLDS[-1])
        assert "send_msg_text" in out

    def test_never_blocks(self):
        """只返回文本提醒，不返回任何拦截信号"""
        for count in range(0, 15):
            assert isinstance(_repeat_reminder(count), str)

    @pytest.mark.parametrize("count", range(1, 15))
    def test_no_unformatted_placeholder_ever_reaches_the_model(self, count: int):
        assert "{" not in _repeat_reminder(count)


class TestThresholdConfig:
    def test_thresholds_are_ascending_and_start_at_two(self):
        assert list(_REPEAT_THRESHOLDS) == sorted(_REPEAT_THRESHOLDS)
        assert _REPEAT_THRESHOLDS[0] >= 2, "阈值 1 会把每一次首轮提交都当成重复"
        assert len(set(_REPEAT_THRESHOLDS)) == len(_REPEAT_THRESHOLDS)


class TestChainSemantics:
    """模拟 run_agent 循环里的计数逻辑"""

    @staticmethod
    def _run(scripts):
        signature, count, fired = "", 0, []
        for script in scripts:
            sig = _canonical_code(script)
            if sig and sig == signature:
                count += 1
            else:
                signature, count = sig, 1
            fired.append(_repeat_reminder(count))
        return fired

    def test_a_different_script_resets_the_chain(self):
        fired = self._run(["a=1", "a=1", "b=2", "b=2"])
        assert fired[0] == ""
        assert fired[1] != ""
        assert fired[2] == "", "换了脚本必须重新计数"
        assert fired[3] != ""

    def test_alternating_scripts_never_fire(self):
        """交替提交两份脚本不算连续重复"""
        assert all(f == "" for f in self._run(["a=1", "b=2"] * 5))

    def test_sustained_repetition_escalates(self):
        fired = self._run(["a=1"] * 8)
        assert fired[0] == ""
        assert fired[_REPEAT_THRESHOLDS[0] - 1] != ""
        assert fired[-1] != ""

    def test_empty_script_does_not_start_a_chain(self):
        """解析不出代码时指纹为空，不应被当作重复提交"""
        assert all(f == "" for f in self._run(["", "", ""]))
