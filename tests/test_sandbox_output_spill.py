"""沙盒输出截断与溢出落盘

回归重点：
1. 输出上限来自配置而不是写死的 1000（原先两个调用方都没传，等于永远 1000）
2. 被截断时完整输出要落到共享目录，并把恢复手段写进截断提示
3. 落盘失败不能把一次成功的执行变成失败
"""

from pathlib import Path

import pytest

from nekro_agent.services.sandbox.runner import (
    SPILL_DIR_NAME,
    SPILL_KEEP_FILES,
    _build_final_output,
    _write_output_spill,
)
from nekro_agent.tools.common_util import limited_text_output


class TestLimitedTextOutput:
    """limited_text_output 的边界"""

    def test_short_text_untouched(self):
        assert limited_text_output("hello", limit=100) == "hello"

    def test_keeps_head_and_tail(self):
        result = limited_text_output("A" * 500 + "B" * 500, limit=100, placeholder="<cut>")
        assert result.startswith("A")
        assert result.endswith("B")
        assert "<cut>" in result

    def test_placeholder_longer_than_limit_still_shrinks(self):
        """占位符比 limit 长时，旧实现 left_limit 变负数，反而几乎原样返回"""
        text = "X" * 10000
        placeholder = "P" * 300
        result = limited_text_output(text, limit=200, placeholder=placeholder)
        assert len(result) < len(text)
        assert len(result) <= 200 + len(placeholder)
        assert placeholder in result

    def test_zero_limit(self):
        result = limited_text_output("X" * 100, limit=0, placeholder="<cut>")
        assert result == "<cut>"

    def test_negative_limit_does_not_expand(self):
        text = "X" * 100
        result = limited_text_output(text, limit=-50, placeholder="<cut>")
        assert len(result) < len(text)


class TestWriteOutputSpill:
    """完整输出落盘"""

    def test_writes_full_text_and_returns_sandbox_path(self, tmp_path: Path):
        payload = "line\n" * 5000
        path = _write_output_spill(tmp_path, payload)

        assert path is not None
        assert path.startswith(f"./shared/{SPILL_DIR_NAME}/")

        written = list((tmp_path / SPILL_DIR_NAME).glob("*.txt"))
        assert len(written) == 1
        assert written[0].read_text(encoding="utf-8") == payload

    def test_path_matches_the_file_actually_written(self, tmp_path: Path):
        """返回的路径必须指向真实文件，否则模型下一轮会 FileNotFoundError"""
        path = _write_output_spill(tmp_path, "payload")
        assert path is not None
        filename = path.rsplit("/", 1)[-1]
        assert (tmp_path / SPILL_DIR_NAME / filename).read_text(encoding="utf-8") == "payload"

    def test_prunes_old_spills(self, tmp_path: Path):
        spill_dir = tmp_path / SPILL_DIR_NAME
        spill_dir.mkdir()
        for i in range(SPILL_KEEP_FILES + 10):
            stale = spill_dir / f"old_{i:03d}.txt"
            stale.write_text("old", encoding="utf-8")

        _write_output_spill(tmp_path, "new payload")

        assert len(list(spill_dir.glob("*.txt"))) <= SPILL_KEEP_FILES + 1

    def test_unwritable_target_returns_none(self, tmp_path: Path):
        """落盘失败要安静降级，不能抛出去毁掉整轮执行"""
        blocked = tmp_path / "shared"
        blocked.write_text("i am a file, not a directory", encoding="utf-8")
        assert _write_output_spill(blocked, "payload") is None


class TestBuildFinalOutput:
    """截断提示与恢复手段"""

    def test_under_limit_passes_through(self, tmp_path: Path):
        assert _build_final_output("short", 1000, tmp_path).text == "short"
        assert not (tmp_path / SPILL_DIR_NAME).exists()

    def test_truncation_names_a_readable_path(self, tmp_path: Path):
        payload = "D" * 40000
        result = _build_final_output(payload, 1000, tmp_path).text

        assert "truncated" in result
        assert f"./shared/{SPILL_DIR_NAME}/" in result
        # 恢复动作必须是可执行的下一步，而不只是"输出太长了"
        assert "exit(9)" in result
        assert "open(" in result

        spilled = list((tmp_path / SPILL_DIR_NAME).glob("*.txt"))
        assert spilled[0].read_text(encoding="utf-8") == payload

    def test_quoted_path_in_hint_is_the_real_file(self, tmp_path: Path):
        """提示里 print(open('...')) 的路径解析回宿主机必须命中真实文件"""
        result = _build_final_output("E" * 40000, 1000, tmp_path).text
        quoted = result.split("open('")[1].split("')")[0]
        assert (tmp_path / quoted.replace("./shared/", "")).exists()

    def test_spill_disabled_says_it_is_unrecoverable(self, tmp_path: Path, monkeypatch):
        from nekro_agent.core.config import config

        monkeypatch.setattr(config, "SANDBOX_OUTPUT_SPILL", False)
        result = _build_final_output("F" * 40000, 1000, tmp_path).text

        assert "NOT recoverable" in result
        assert not (tmp_path / SPILL_DIR_NAME).exists()

    def test_hidden_count_is_accurate(self, tmp_path: Path):
        payload = "G" * 5000
        result = _build_final_output(payload, 1000, tmp_path).text
        assert f"{5000 - 1000} of 5000 characters hidden" in result

    @pytest.mark.parametrize("limit", [0, 1, 50, 1000, 8192])
    def test_never_expands_the_output(self, tmp_path: Path, limit: int):
        """任何上限下，截断结果都不能比原文还长"""
        payload = "H" * 60000
        result = _build_final_output(payload, limit, tmp_path).text
        assert len(result) < len(payload)


class TestTruncationIsReportedIndependently:
    """截断与退出类型正交：exit 0 的成功执行也可能只让模型看到了 2% 的输出"""

    def test_untruncated_run_reports_no_truncation(self, tmp_path: Path):
        result = _build_final_output("short output", 1000, tmp_path)
        assert result.truncated is False
        assert result.total_chars == len("short output")
        assert result.spill_path == ""

    def test_truncated_run_reports_original_size(self, tmp_path: Path):
        result = _build_final_output("I" * 40000, 1000, tmp_path)
        assert result.truncated is True
        assert result.total_chars == 40000
        assert result.spill_path.startswith(f"./shared/{SPILL_DIR_NAME}/")

    def test_total_chars_is_the_pre_truncation_length(self, tmp_path: Path):
        """记的必须是截断前的长度，否则事后分不出输出本来就短还是被砍了"""
        result = _build_final_output("J" * 12345, 1000, tmp_path)
        assert result.total_chars == 12345
        assert len(result.text) < result.total_chars

    def test_spill_path_empty_when_spill_disabled(self, tmp_path: Path, monkeypatch):
        from nekro_agent.core.config import config

        monkeypatch.setattr(config, "SANDBOX_OUTPUT_SPILL", False)
        result = _build_final_output("K" * 40000, 1000, tmp_path)
        assert result.truncated is True
        assert result.spill_path == ""

    def test_ext_data_defaults_are_backwards_compatible(self):
        """老记录的 extra_data 里没有这几个字段，反序列化必须落到未截断"""
        from nekro_agent.schemas.sandbox import SandboxCodeExtData

        legacy = {
            "message_cnt": 1,
            "token_consumption": 10,
            "token_input": 5,
            "token_output": 5,
            "chars_count_input": 100,
            "chars_count_output": 50,
            "chars_count_total": 150,
            "use_model": "test",
            "speed_tokens_per_second": 1.0,
            "speed_chars_per_second": 1.0,
            "first_token_cost_ms": 1,
            "generation_time_ms": 1,
            "stream_mode": False,
        }
        parsed = SandboxCodeExtData(**legacy)
        assert parsed.output_truncated is False
        assert parsed.output_chars_total == 0
        assert parsed.output_spill_path == ""
