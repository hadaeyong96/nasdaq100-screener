"""종목별 다음 실적 발표일을 yfinance로 받아 하루 단위로 캐시한다.

실패하거나 알 수 없으면 None을 반환하고, 전체 실행은 멈추지 않는다
(core/filters.py의 실적 필터는 None이면 earnings_unknown=True로만 표시한다).
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent
CACHE_PATH = DATA_DIR / "cache" / "earnings.parquet"


def _fetch_next_earnings_date(ticker: str, today: date) -> date | None:
    """yfinance에서 이 종목의 다음(오늘 이후) 실적 발표일을 찾는다.

    yfinance 버전에 따라 API 형태가 달라질 수 있어, 실패하면 조용히 None.
    """
    try:
        t = yf.Ticker(ticker)
        df = t.get_earnings_dates(limit=12)
        if df is None or df.empty:
            return None
        candidates = [d.date() for d in df.index if d.date() >= today]
        return min(candidates) if candidates else None
    except Exception:
        return None


def _load_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame(columns=["ticker", "earnings_date", "fetched_on"]).set_index("ticker")
    df = pd.read_parquet(cache_path)
    return df


def _save_cache(df: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)


def get_earnings_dates(
    tickers: list[str], today: date | None = None, cache_path: Path = CACHE_PATH
) -> dict[str, date | None]:
    """종목별 다음 실적 발표일을 받는다 (하루 단위 캐시, cache_path에 저장).

    입력: yfinance 형식 ticker 목록, today(테스트용 날짜 주입), cache_path
    출력: {ticker: date 또는 None}
    """
    today = today or datetime.now().date()
    cache = _load_cache(cache_path)

    result: dict[str, date | None] = {}
    updated = False
    for ticker in tickers:
        cached_row = cache.loc[ticker] if ticker in cache.index else None
        if cached_row is not None and pd.Timestamp(cached_row["fetched_on"]).date() == today:
            raw = cached_row["earnings_date"]
            result[ticker] = None if pd.isna(raw) else pd.Timestamp(raw).date()
            continue

        fetched = _fetch_next_earnings_date(ticker, today)
        result[ticker] = fetched
        cache.loc[ticker] = {"earnings_date": fetched, "fetched_on": today}
        updated = True

    if updated:
        _save_cache(cache, cache_path)
    return result
