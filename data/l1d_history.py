"""L1d 해외 지수·단기 금리 로더 (docs/l1d_plan.md 데이터 절).

- 지수: yfinance auto_adjust=False Close, 1950-01-01 ~ 2022-01-01(끝 제외), data/cache/l1d/<티커>.csv에 캐시.
- 금리: FRED 월별(OECD) 시리즈와 DTB3(일별 → 월평균), data/cache/l1d/fred_<ID>.csv에 캐시.
모든 결과는 2021-12-31까지 자르고 봉인 검사(core.l1d.cut_and_seal)를 거친다. 네트워크 실패는 예외로 멈춘다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core import l1d

CACHE = Path(__file__).resolve().parent / "cache" / "l1d"


def _safe(name: str) -> str:
    return name.replace("^", "")


def load_index(ticker: str) -> tuple[pd.Series, dict]:
    """지수 종가(정리·봉인 후)와 정리 기록. 캐시가 없으면 받는다. 실패하면 RuntimeError."""
    path = CACHE / f"{_safe(ticker)}.csv"
    if path.exists():
        raw = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        import yfinance as yf

        df = yf.download(ticker, start="1950-01-01", end="2022-01-01", auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            raise RuntimeError(f"{ticker} 내려받기 실패(빈 결과)")
        close = df["Close"]
        raw = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
        raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
        CACHE.mkdir(parents=True, exist_ok=True)
        raw.rename("close").to_csv(path)
    raw = l1d.cut_and_seal(raw.astype(float), ticker)
    s, info = l1d.clean_index(raw)
    s.name = "close"
    return s, info


def load_fred_monthly(series_id: str) -> pd.Series:
    """FRED 시리즈를 월별 연율 %로(DTB3는 일별 → 그달 평균). 인덱스 = 월 Period. 실패하면 예외."""
    path = CACHE / f"fred_{series_id}.csv"
    if path.exists():
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        from data import fx

        raw = fx.fetch_fred_series_range(series_id, "1950-01-01", "2021-12-31")
        if not raw:
            raise RuntimeError(f"FRED {series_id} 빈 결과")
        s = pd.Series(raw, dtype=float)
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        CACHE.mkdir(parents=True, exist_ok=True)
        s.rename(series_id).to_csv(path)
    s = l1d.cut_and_seal(s.astype(float), series_id)
    return s.groupby(s.index.to_period("M")).mean()
