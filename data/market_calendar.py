"""미국 증시(NYSE) 거래일 달력 (P3.2).

`pandas_market_calendars`는 라이브러리에 내장된 캘린더 규칙으로 스케줄을
계산하므로 네트워크를 쓰지 않는다. 주말·공휴일·조기 폐장(반나절장)을 정확히
반영해 "실행 시각 기준 가장 최근에 마감된 거래일"을 계산한다. 기존
`data/prices.py`의 주말만 건너뛰는 근사치와 config.yaml의 공휴일 TODO를
대체한다 (P3.2 지시문 2번).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pandas_market_calendars as mcal

_NYSE = mcal.get_calendar("NYSE")
# 연휴가 이어져도(예: 추수감사절 연휴) 충분히 거슬러 올라가도록 여유를 둔다.
_LOOKBACK_DAYS = 15


def latest_closed_trading_day(now_et: datetime) -> date:
    """실행 시각(미국 동부) 기준 가장 최근에 정규장이 마감된 거래일을 구한다.

    입력: now_et(미국 동부 시각, tz-aware datetime)
    출력: date. NYSE 캘린더로 주말·공휴일·조기 폐장을 정확히 반영한다
         (조기 폐장일은 그날의 실제 마감 시각을 기준으로 판정한다).
    예외: 조회 구간(_LOOKBACK_DAYS일) 안에 마감된 거래일이 없으면 ValueError
         (정상 상황에서는 발생하지 않는다).
    """
    start = (now_et - timedelta(days=_LOOKBACK_DAYS)).date()
    end = now_et.date()
    schedule = _NYSE.schedule(start_date=start, end_date=end)
    now_ts = pd.Timestamp(now_et)
    closed = schedule[schedule["market_close"] <= now_ts]
    if closed.empty:
        raise ValueError(f"{_LOOKBACK_DAYS}일 안에 마감된 거래일을 찾지 못했습니다 (now_et={now_et.isoformat()})")
    return closed.index[-1].date()
