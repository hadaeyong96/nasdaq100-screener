"""나스닥 100 종목의 일봉 시세를 yfinance로 받아 캐시한다.

- `auto_adjust=False`의 `Close`(분할 반영, 배당 미반영)만 쓴다.
- 미국 동부 시각 기준으로 아직 장이 끝나지 않은 당일 봉은 제거한다.
- `data/cache/{ticker}.parquet`에 캐시하고, 캐시가 오늘 확정 봉까지 있으면
  다시 받지 않는다.
- 실패 종목은 모아서 반환하고 전체 실행은 계속한다.
- 상장 기간이 짧은 종목(최근 IPO 등)은 실패로 처리하지 않는다. 확보한 거래일이
  history_days보다 적어도 그대로 반환하고, core.indicators의 bars/
  insufficient_history로 부족 여부를 표시한다 (P1.1 3번).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
# YF_DISABLE_CURL_CFFI 같은 네트워크 우회 설정을 .env에서 읽는다.
# yfinance는 import 시점에 이 환경변수를 읽으므로, yfinance를 import하기 전에
# 반드시 먼저 .env를 로드해야 한다 (학교·회사 네트워크 TLS 우회, P1.1 4번).
load_dotenv(ROOT / ".env")

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / "cache"

US_EASTERN = ZoneInfo("America/New_York")
# 미국 정규장 마감 시각 (동부시간). 조기 폐장일(반나절장)은 반영하지 않는다 — [판단 필요].
# 미국 공휴일 캘린더도 아직 반영하지 않는다 — P4(스케줄)에서 처리 예정 (P1.1 6번).
MARKET_CLOSE_HOUR = 16

_RAW_TO_LOWER = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
_PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
_CORE_PRICE_COLUMNS = ["open", "high", "low", "close"]  # volume은 별도 취급 (P1.1 1번)


@dataclass
class PriceFetchResult:
    """여러 종목 시세 수집 결과.

    prices: {ticker: DataFrame(open, high, low, close, volume, ...)}
    failed: {ticker: 실패 사유 문자열}
    warnings: {ticker: [경고 문자열, ...]} — 실패는 아니지만 확인이 필요한 경우
    """

    prices: dict = field(default_factory=dict)
    failed: dict = field(default_factory=dict)
    warnings: dict = field(default_factory=dict)


def _latest_confirmed_trading_date(now_et: datetime) -> date:
    """이 시각 기준으로 확정됐을 것으로 기대하는 가장 최근 거래일을 구한다.

    입력: 미국 동부 시각 datetime
    출력: date. 주말은 건너뛰지만 미국 공휴일 캘린더는 반영하지 않는다
         (P4에서 처리 예정 — P1.1 6번).
    """
    d = now_et.date()
    if now_et.hour < MARKET_CLOSE_HOUR:
        d -= timedelta(days=1)
    while d.weekday() >= 5:  # 토(5) · 일(6)
        d -= timedelta(days=1)
    return d


def _period_for(history_days: int) -> str:
    """history_days 거래일을 넉넉히 확보하기 위한 yfinance period 문자열을 만든다."""
    years = max(2, math.ceil(history_days / 250) + 1)
    return f"{years}y"


def _clean_raw(raw: pd.DataFrame, now_et: datetime) -> pd.DataFrame:
    """yfinance raw OHLCV를 정리한다.

    입력: yfinance Ticker.history(auto_adjust=False) 반환 DataFrame (tz-aware 인덱스)
    출력: DataFrame(open, high, low, close, volume), 날짜 오름차순, 확정 봉만.
         out.attrs["warnings"]에 실패는 아니지만 확인이 필요한 메시지를 담는다.
    """
    out = raw.rename(columns=_RAW_TO_LOWER)[_PRICE_COLUMNS].copy()
    out.index = out.index.tz_convert(US_EASTERN)
    out = out.sort_index()

    if len(out) and now_et.hour < MARKET_CLOSE_HOUR and out.index[-1].date() == now_et.date():
        out = out.iloc[:-1]  # 미국장이 아직 안 끝난 당일 봉 제거

    # 마감 직후(또는 야후 데이터 파이프라인 지연)에는 종가 등 일부 값이 NaN인 채로
    # 내려오는 경우가 있다. open/high/low/close 중 하나라도 NaN이면 확정되지 않은
    # 값으로 보고 마지막 봉들을 제거한다. volume만 NaN이거나 0인 경우는 가격이
    # 이미 확정된 것이므로 버리지 않고 경고만 남긴다 (P1.1 1번).
    while len(out) and out.iloc[-1][_CORE_PRICE_COLUMNS].isna().any():
        out = out.iloc[:-1]

    warnings: list[str] = []
    if len(out):
        last_volume = out.iloc[-1]["volume"]
        if pd.isna(last_volume) or last_volume == 0:
            last_date = out.index[-1].date().isoformat()
            warnings.append(f"{last_date} 거래량이 NaN 또는 0 (봉은 유지, 확인 필요)")

    out.index = out.index.tz_localize(None).normalize()
    out.index.name = "date"
    out.attrs["warnings"] = warnings
    return out


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.parquet"


def _load_cache(ticker: str) -> pd.DataFrame | None:
    """캐시 파일을 읽는다. 없거나 손상됐으면 None."""
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


def _cache_is_fresh(cached: pd.DataFrame | None, now_et: datetime) -> bool:
    """캐시 마지막 날짜가 오늘 기대하는 확정 거래일까지 있으면 True.

    history_days 기준 최소 보유량은 더 이상 신선도 조건에 넣지 않는다.
    상장 기간이 짧은 종목(P1.1 3번)은 이미 상장일 이후 전 구간을 캐시에
    가지고 있어도 history_days에 못 미칠 수 있는데, 이 경우 매번 재요청해도
    더 받아올 데이터가 없기 때문이다.
    """
    if cached is None or cached.empty:
        return False
    latest_needed = _latest_confirmed_trading_date(now_et)
    return cached.index[-1].date() >= latest_needed


def fetch_one(ticker: str, cfg: dict, now_et: datetime | None = None) -> pd.DataFrame:
    """종목 하나의 확정 일봉을 받는다 (캐시 우선).

    입력: yfinance 형식 ticker, cfg(config.yaml 로드값), now_et(테스트용 시각 주입)
    출력: DataFrame(open, high, low, close, volume), 날짜 오름차순 인덱스.
         out.attrs["warnings"]에 실패는 아니지만 확인이 필요한 메시지 목록.
    예외: 확정된 가격 데이터가 전혀 없으면 ValueError
         (history_days 미만이어도 상장 기간이 짧으면 실패로 보지 않는다 — P1.1 3번)
    """
    history_days = cfg["data"]["history_days"]
    now_et = now_et or datetime.now(US_EASTERN)

    cached = _load_cache(ticker)
    if _cache_is_fresh(cached, now_et):
        cached.attrs.setdefault("warnings", [])
        return cached

    raw = yf.Ticker(ticker).history(period=_period_for(history_days), auto_adjust=False)
    if raw.empty:
        raise ValueError("가격 데이터 없음")

    cleaned = _clean_raw(raw, now_et)
    if cleaned.empty:
        raise ValueError("확정된 가격 데이터 없음 (전부 NaN)")

    warnings = cleaned.attrs.get("warnings", [])
    if len(cleaned) < history_days:
        warnings.append(
            f"거래일 {len(cleaned)}일 < history_days({history_days}) - 상장 기간이 짧은 종목일 수 있음"
        )

    latest_needed = _latest_confirmed_trading_date(now_et)
    if cleaned.index[-1].date() < latest_needed:
        warnings.append(
            f"기대한 최신 확정 거래일({latest_needed.isoformat()})보다 오래된 데이터"
            f"(마지막={cleaned.index[-1].date().isoformat()}). 야후 파이낸스 데이터 반영 지연 가능성."
        )

    cleaned.attrs["warnings"] = warnings
    _save_cache(ticker, cleaned)
    return cleaned


def fetch_universe_prices(tickers: list[str], cfg: dict, sleep_sec: float = 0.3) -> PriceFetchResult:
    """여러 종목의 확정 일봉을 받는다. 실패 종목은 모아서 반환하고 계속 진행한다.

    입력: yfinance 형식 ticker 목록, cfg(config.yaml 로드값)
    출력: PriceFetchResult(prices, failed, warnings)
    """
    result = PriceFetchResult()
    now_et = datetime.now(US_EASTERN)
    for ticker in tickers:
        try:
            df = fetch_one(ticker, cfg, now_et=now_et)
            result.prices[ticker] = df
            msgs = df.attrs.get("warnings") or []
            if msgs:
                result.warnings[ticker] = msgs
        except Exception as exc:
            result.failed[ticker] = str(exc)
        time.sleep(sleep_sec)
    return result


if __name__ == "__main__":
    import sys

    import yaml

    for _stream in (sys.stdout, sys.stderr):  # 윈도우 콘솔 cp949 UnicodeEncodeError 방지
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")

    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)
    _res = fetch_universe_prices(["AAPL", "NVDA"], _cfg)
    for _t, _df in _res.prices.items():
        print(_t, len(_df), _df.index[-1].date())
    print("경고:", _res.warnings)
    print("실패:", _res.failed)
