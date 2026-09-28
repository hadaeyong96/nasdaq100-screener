"""종목별 GICS 섹터를 yfinance로 받아 캐시한다 (KJB-1, docs/kjb_nasdaq100_plan.md 2장 #6).

섹터는 시점별로 다시 받지 않고 "현재" 분류를 전 기간에 그대로 쓴다 — 과거 실제
분류와 다를 수 있음을 안다(계획서 6장 한계). 그래서 실적일(data/earnings.py)과
달리 하루 단위로 갱신하지 않고, 한 번 받으면 캐시에 영구히 남긴다(수동으로
지우기 전까지). 실패한 종목은 캐시에 None으로 남겨 다음에도 계속 재시도하지
않는다 — 실패 목록은 반환값으로 보고한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent
CACHE_PATH = DATA_DIR / "cache" / "sectors.parquet"


def _fetch_sector(ticker: str) -> str | None:
    """yfinance Ticker.info에서 GICS 섹터를 받는다. 실패하면 조용히 None."""
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector")
        return sector if sector else None
    except Exception:
        return None


def _load_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame(columns=["ticker", "sector"]).set_index("ticker")
    return pd.read_parquet(cache_path)


def _save_cache(df: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)


def get_sectors(tickers: list[str], cache_path: Path = CACHE_PATH) -> tuple[dict[str, str], list[str]]:
    """종목별 GICS 섹터를 받는다 (영구 캐시, cache_path에 저장).

    입력: yfinance 형식 ticker 목록, cache_path
    출력: ({ticker: 섹터명}(받은 것만), 실패한 ticker 목록)
    """
    cache = _load_cache(cache_path)
    result: dict[str, str] = {}
    failed: list[str] = []
    updated = False

    for ticker in tickers:
        if ticker in cache.index and pd.notna(cache.loc[ticker, "sector"]):
            result[ticker] = cache.loc[ticker, "sector"]
            continue
        if ticker in cache.index:
            failed.append(ticker)  # 예전에 이미 실패로 캐시된 종목 — 다시 시도하지 않는다
            continue
        sector = _fetch_sector(ticker)
        cache.loc[ticker, "sector"] = sector
        updated = True
        if sector is None:
            failed.append(ticker)
        else:
            result[ticker] = sector

    if updated:
        _save_cache(cache, cache_path)

    return result, failed
