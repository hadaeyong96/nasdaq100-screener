"""P3.2 2번: NYSE 거래일 달력 기준 가장 최근 마감 거래일 계산 테스트.

pandas_market_calendars는 네트워크 없이 내장 규칙으로 스케줄을 계산하므로
이 테스트도 네트워크 없이 돈다.
"""

from __future__ import annotations

from datetime import date, datetime

from data.market_calendar import latest_closed_trading_day
from data.prices import US_EASTERN


def test_weekday_before_close_returns_previous_trading_day():
    """정규장 마감(16:00 ET) 전이면 아직 오늘 봉이 확정되지 않았다."""
    now_et = datetime(2026, 9, 23, 7, 0, tzinfo=US_EASTERN)  # 수요일 아침
    assert latest_closed_trading_day(now_et) == date(2026, 9, 22)


def test_weekday_after_close_returns_today():
    now_et = datetime(2026, 9, 23, 20, 0, tzinfo=US_EASTERN)  # 수요일 밤(마감 후)
    assert latest_closed_trading_day(now_et) == date(2026, 9, 23)


def test_monday_before_open_skips_weekend_to_friday():
    now_et = datetime(2026, 9, 21, 7, 0, tzinfo=US_EASTERN)  # 월요일 아침
    assert latest_closed_trading_day(now_et) == date(2026, 9, 18)  # 지난 금요일


def test_day_after_holiday_skips_holiday():
    """추수감사절(2026-11-26, 목) 다음 날 아침에는 그 전 마감 거래일(11/25)을 가리킨다."""
    now_et = datetime(2026, 11, 27, 7, 0, tzinfo=US_EASTERN)  # 추수감사절 다음 날(휴장) 아침
    assert latest_closed_trading_day(now_et) == date(2026, 11, 25)


def test_early_close_day_uses_actual_close_time():
    """추수감사절 다음 날(블랙프라이데이)은 조기 폐장(13:00 ET)이다.
    조기 폐장 시각 이후면 그날이 마감 거래일로 잡혀야 한다."""
    now_et = datetime(2026, 11, 27, 14, 0, tzinfo=US_EASTERN)  # 조기 폐장(13:00 ET) 이후
    assert latest_closed_trading_day(now_et) == date(2026, 11, 27)
