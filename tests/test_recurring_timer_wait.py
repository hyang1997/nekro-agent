"""Recurring-timer wait bounding and misfire arithmetic.

The morning brief silently stopped firing. The jobs were scheduled correctly — the loop had
simply converted "next run" into a single ~16-hour asyncio timeout that never expired, so both
timers sat due-but-untouched in the heap until an unrelated call set() the wakeup event and
fired them hours late. A one-minute job on the same loop fired fine, which is what pinned the
fault on the length of the wait rather than the loop.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from nekro_agent.services.timer.recurring_timer_service import (
    _MAX_WAIT_SECONDS,
    _MISFIRE_TOLERANCE_SECONDS,
)

TORONTO = ZoneInfo("America/Toronto")
UTC = ZoneInfo("UTC")


class TestWaitBounding:
    @staticmethod
    def _wait(due_in: float) -> float:
        return min(due_in, _MAX_WAIT_SECONDS)

    def test_overnight_wait_is_chunked(self):
        """07:55 armed at 15:36 the day before is ~16h; that must not become one timeout."""
        assert self._wait(58_740) == _MAX_WAIT_SECONDS

    def test_imminent_wait_is_not_extended(self):
        assert self._wait(3.5) == 3.5

    def test_wait_never_exceeds_the_cap(self):
        for due_in in (0.001, 1, 59, 60, 61, 3600, 86_400, 604_800):
            assert self._wait(due_in) <= _MAX_WAIT_SECONDS

    def test_cap_bounds_worst_case_lateness(self):
        """A missed wakeup costs at most one cap, not the whole remaining interval."""
        assert _MAX_WAIT_SECONDS <= 60.0

    def test_repeated_waits_reach_a_distant_deadline(self):
        remaining, waits = 58_740.0, 0
        while remaining > 0:
            remaining -= self._wait(remaining)
            waits += 1
            assert waits < 2000, "chunked waiting failed to converge"
        assert remaining <= 0


class TestMisfireArithmetic:
    """`.replace(tzinfo=...)` relabels instead of converting, shifting the comparison by the
    zone's UTC offset. Tortoise hands back an aware datetime, so this has to convert."""

    @staticmethod
    def _lag(next_run_at: datetime, fired_at: datetime) -> float:
        return (fired_at - next_run_at.astimezone(fired_at.tzinfo)).total_seconds()

    def test_on_time_fire_has_near_zero_lag(self):
        due = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
        fired = due.astimezone(TORONTO) + timedelta(seconds=4)
        assert self._lag(due, fired) == pytest.approx(4, abs=1)

    def test_replace_would_have_been_wrong_by_the_utc_offset(self):
        due = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
        fired = due.astimezone(TORONTO) + timedelta(seconds=4)
        wrong = (fired - due.replace(tzinfo=fired.tzinfo)).total_seconds()
        assert abs(wrong - self._lag(due, fired)) == pytest.approx(4 * 3600, abs=1)

    def test_genuinely_late_fire_is_a_misfire(self):
        due = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
        fired = due.astimezone(TORONTO) + timedelta(hours=4)
        assert self._lag(due, fired) > _MISFIRE_TOLERANCE_SECONDS

    def test_normal_jitter_is_not_a_misfire(self):
        due = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
        for jitter in (0, 1, 30, 60):
            fired = due.astimezone(TORONTO) + timedelta(seconds=jitter)
            assert self._lag(due, fired) <= _MISFIRE_TOLERANCE_SECONDS

    def test_tolerance_exceeds_the_wait_cap(self):
        """Otherwise a fire delayed by one ordinary wait chunk is mislabelled a makeup."""
        assert _MISFIRE_TOLERANCE_SECONDS > _MAX_WAIT_SECONDS

    def test_lag_is_offset_independent(self):
        due = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
        fired_utc = due + timedelta(seconds=10)
        assert self._lag(due, fired_utc) == pytest.approx(
            self._lag(due, fired_utc.astimezone(TORONTO)), abs=0.001,
        )
