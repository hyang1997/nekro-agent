"""历史窗口裁剪

原实现有两个问题：丢弃是完全静默的（模型分不清"对话刚开始"和"刚被截掉四十轮"），
以及单条消息超预算时会把整段历史清空。
"""

from nekro_agent.services.agent.templates.history import _trim_history_to_budget


class TestTrimHistoryToBudget:
    def test_everything_fits(self):
        prompts = ["a" * 10, "b" * 10, "c" * 10]
        kept, omitted = _trim_history_to_budget(prompts, 100)
        assert kept == prompts
        assert omitted == 0

    def test_empty_input(self):
        assert _trim_history_to_budget([], 100) == ([], 0)

    def test_drops_oldest_first(self):
        prompts = ["old" * 10, "mid" * 10, "new" * 10]
        kept, omitted = _trim_history_to_budget(prompts, 60)
        assert omitted == 1
        assert kept == prompts[1:]
        assert kept[-1] == prompts[-1]

    def test_omitted_count_matches_what_was_removed(self):
        prompts = [f"msg{i:03d}" * 5 for i in range(50)]
        kept, omitted = _trim_history_to_budget(prompts, 500)
        assert omitted + len(kept) == len(prompts)
        assert kept == prompts[omitted:]

    def test_newest_message_always_survives(self):
        """单条就超预算时不能清空历史——那等于让模型对着空白回答刚收到的消息"""
        prompts = ["short", "X" * 100000]
        kept, omitted = _trim_history_to_budget(prompts, 100)
        assert len(kept) == 1
        assert kept[0] == prompts[-1]
        assert omitted == 1

    def test_single_oversized_message_alone(self):
        prompts = ["Y" * 100000]
        kept, omitted = _trim_history_to_budget(prompts, 10)
        assert kept == prompts
        assert omitted == 0

    def test_zero_budget_keeps_exactly_one(self):
        prompts = ["a", "b", "c"]
        kept, omitted = _trim_history_to_budget(prompts, 0)
        assert kept == ["c"]
        assert omitted == 2

    def test_budget_boundary_is_inclusive(self):
        prompts = ["a" * 30, "b" * 30]
        kept, _ = _trim_history_to_budget(prompts, 60)
        assert len(kept) == 2

    def test_one_over_budget_drops_one(self):
        prompts = ["a" * 31, "b" * 30]
        kept, omitted = _trim_history_to_budget(prompts, 60)
        assert omitted == 1
        assert kept == prompts[1:]

    def test_kept_is_always_a_contiguous_suffix(self):
        prompts = [f"m{i}" * (i + 1) for i in range(30)]
        for budget in (0, 1, 5, 50, 200, 100000):
            kept, omitted = _trim_history_to_budget(prompts, budget)
            assert kept == prompts[omitted:], f"budget={budget} 保留的不是连续后缀"
            assert kept, f"budget={budget} 把历史清空了"
