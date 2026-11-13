"""실적급등 매도 조건 A의 다음 실적 발표 예정일 — yfinance earnings_dates (docs/prereg/pead_exits.md 1-1).

실패는 예외로 올린다. 엔진(engine/pead_exits)이 8-K 간격 추정·기본 60일로 넘어가고 사유를 기록한다.
"""

from __future__ import annotations

import pandas as pd


def earnings_dates(ticker: str) -> list[str]:
    """yfinance Ticker.get_earnings_dates(limit=12)의 날짜들('YYYY-MM-DD', 오름차순). 표가 비면 빈 목록."""
    import yfinance as yf

    df = yf.Ticker(ticker).get_earnings_dates(limit=12)
    if df is None or df.empty:
        return []
    return sorted({pd.Timestamp(x).date().isoformat() for x in df.index})
