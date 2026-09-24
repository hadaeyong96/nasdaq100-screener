"""백테스트용 과거 일봉 수집 (P5-1 2번).

라이브(data/prices.py)와 달리 "지금" 시각에 의존하지 않고 고정된 [start, end] 구간을
받는다. 데이터 규칙은 라이브와 같다(분할 반영 Close, `auto_adjust=False`). 누락 거래일
복구·data_gap 규칙도 라이브와 같은 순수 함수(data.prices.find_mid_series_gaps,
data.prices.recover_gap_days)를 그대로 쓴다 — "당일 아직 안 끝난 봉 트림", "meta API로
당일 실시간 종가 보완" 부분만 뺀다(과거 데이터라 필요 없다).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from data.prices import CLOSE_SOURCE_YAHOO, find_mid_series_gaps, recover_gap_days

CACHE_DIR = Path(__file__).resolve().parent / "cache" / "backtest"
_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class BacktestPriceResult:
    """여러 종목 과거 일봉 수집 결과.

    prices: {ticker: DataFrame(open, high, low, close, volume, close_source)}
    failed: {ticker: 실패 사유}
    data_gap: {ticker: [끝내 복구 못 한 날짜, ...]}
    """

    prices: dict = field(default_factory=dict)
    failed: dict = field(default_factory=dict)
    data_gap: dict = field(default_factory=dict)


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.parquet"


def _load_cache(ticker: str) -> pd.DataFrame | None:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_cache_path(ticker))


def _fetch_raw(ticker: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.Ticker(ticker).history(start=start, end=end + timedelta(days=1), auto_adjust=False)
    if raw.empty:
        return pd.DataFrame(columns=_COLUMNS + ["close_source"])
    out = raw.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )[_COLUMNS].copy()
    out.index = out.index.tz_localize(None).normalize()
    out.index.name = "date"
    out = out.sort_index()
    out["close_source"] = CLOSE_SOURCE_YAHOO
    while len(out) and out.iloc[-1][["open", "high", "low", "close"]].isna().any():
        out = out.iloc[:-1]  # 방어적: 과거 구간이라도 끝자락에 확정 안 된 값이 섞여 오면 버린다
    return out


def fetch_history(ticker: str, start: date, end: date) -> tuple[pd.DataFrame, list[str]]:
    """ticker의 [start, end] 확정 일봉을 받는다 (캐시가 그 구간을 덮으면 재사용).

    입력: yfinance 형식 ticker, start, end
    출력: (DataFrame(open,high,low,close,volume,close_source), 경고 목록)
    예외: 확정 데이터가 전혀 없으면 ValueError
    """
    warnings: list[str] = []
    cached = _load_cache(ticker)
    if cached is not None and len(cached) and cached.index[0].date() <= start and cached.index[-1].date() >= end:
        out = cached.loc[str(start) : str(end)].copy()
    else:
        out = _fetch_raw(ticker, start, end)
        if out.empty:
            raise ValueError(f"{ticker}: 가격 데이터 없음")
        if cached is not None and len(cached):
            out = pd.concat([cached, out]).sort_index()
            out = out[~out.index.duplicated(keep="last")]
        _save_cache(ticker, out)
        out = out.loc[str(start) : str(end)].copy()

    if out.empty:
        raise ValueError(f"{ticker}: [{start}, {end}] 구간 데이터 없음")

    # 라이브와 같은 순수 함수로 중간 구멍을 찾아 복구한다 (당일 미확정 봉 트림·meta
    # 보완은 과거 데이터에 해당하지 않아 뺀다).
    gap_dates = find_mid_series_gaps(out)
    if gap_dates:
        def _daily(target_date: date):
            row_df = _fetch_raw(ticker, target_date, target_date)
            if row_df.empty:
                return None
            row = row_df.iloc[0]
            return {"open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]),
                    "close": float(row["close"]), "volume": float(row.get("volume") or 0)}

        out, recovered, unresolved, recovery_warnings = recover_gap_days(out, gap_dates, _daily, lambda d: None)
        warnings.extend(recovery_warnings)
        if unresolved:
            warnings.append(f"{ticker}: 끝내 복구 못 한 거래일 {len(unresolved)}개 - data_gap 처리")
        out.attrs["data_gap_dates"] = unresolved
    else:
        out.attrs["data_gap_dates"] = []

    return out, warnings


def fetch_universe_history(tickers: list[str], start: date, end: date) -> BacktestPriceResult:
    """여러 종목의 과거 일봉을 받는다. 실패 종목은 모아서 반환하고 계속 진행한다.

    입력: yfinance 형식 ticker 목록, start, end
    출력: BacktestPriceResult
    """
    result = BacktestPriceResult()
    for ticker in tickers:
        try:
            df, warnings = fetch_history(ticker, start, end)
            result.prices[ticker] = df
            gap_dates = df.attrs.get("data_gap_dates") or []
            if gap_dates:
                result.data_gap[ticker] = [d.date().isoformat() for d in gap_dates]
        except Exception as exc:
            result.failed[ticker] = str(exc)
    return result


def fetch_dividends(ticker: str, start: date, end: date) -> pd.Series:
    """ticker의 [start, end] 배당 이력(주당 배당금)을 받는다. 실패하면 빈 Series.

    출력: pd.Series(index=배당락일, value=주당 배당금(달러))
    """
    import yfinance as yf

    try:
        div = yf.Ticker(ticker).dividends
    except Exception:
        return pd.Series(dtype=float)
    if div is None or div.empty:
        return pd.Series(dtype=float)
    div = div.copy()
    div.index = pd.to_datetime(div.index).tz_localize(None).normalize()
    return div.loc[str(start) : str(end)]
