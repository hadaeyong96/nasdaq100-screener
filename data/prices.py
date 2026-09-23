"""나스닥 100 종목의 일봉 시세를 yfinance로 받아 캐시한다.

- `auto_adjust=False`의 `Close`(분할 반영, 배당 미반영)만 쓴다.
- 미국 동부 시각 기준으로 아직 장이 끝나지 않은 당일 봉은 제거한다.
- `data/cache/{ticker}.parquet`에 캐시하고, 캐시가 오늘 확정 봉까지 있으면
  다시 받지 않는다.
- 실패 종목은 모아서 반환하고 전체 실행은 계속한다.
- 상장 기간이 짧은 종목(최근 IPO 등)은 실패로 처리하지 않는다. 확보한 거래일이
  history_days보다 적어도 그대로 반환하고, core.indicators의 bars/
  insufficient_history로 부족 여부를 표시한다 (P1.1 3번).
- 야후가 가장 최근 거래일의 close만 비워둔 채 내려주는 경우가 있다. 정규장이
  끝난 것이 확인되면 chart API의 `meta.regularMarketPrice`로 채운다 (P1.2 2번).
- meta로 채운 close는 다음 실행에서 실제 일봉 종가가 들어오면 그 값으로 덮어쓴다
  (P2 P1 마무리 1번). 마지막 봉이 아닌 곳에 close 구멍이 남으면 보간하지 않고
  `data_gap_dates`로 표시한다 (P2 P1 마무리 2번).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
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

# close 출처 표시 (P1.2 2번). "yahoo" = yfinance가 준 값 그대로,
# "meta" = chart API의 meta.regularMarketPrice로 채운 값.
_CLOSE_SOURCE_COLUMN = "close_source"
CLOSE_SOURCE_YAHOO = "yahoo"
CLOSE_SOURCE_META = "meta"

CHART_API_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_CHART_API_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class ChartMeta:
    """야후 chart API 응답의 meta 중 close 보완에 필요한 값만 담는다.

    price: meta.regularMarketPrice (해당 종목의 최근 정규장 가격)
    time_et: meta.regularMarketTime을 미국 동부 시각으로 바꾼 값
    """

    price: float | None = None
    time_et: datetime | None = None


@dataclass
class PriceFetchResult:
    """여러 종목 시세 수집 결과.

    prices: {ticker: DataFrame(open, high, low, close, volume, close_source, ...)}
    failed: {ticker: 실패 사유 문자열}
    warnings: {ticker: [경고 문자열, ...]} — 실패는 아니지만 확인이 필요한 경우
    close_filled: close를 meta로 채운 티커 목록 (P1.2 2번)
    data_gap: {ticker: [구멍 날짜, ...]} — 마지막 봉이 아닌 곳에 close가 빈 종목 (P2 P1 마무리 2번)
    """

    prices: dict = field(default_factory=dict)
    failed: dict = field(default_factory=dict)
    warnings: dict = field(default_factory=dict)
    close_filled: list = field(default_factory=list)
    data_gap: dict = field(default_factory=dict)


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


def _needs_close_fill(df: pd.DataFrame) -> bool:
    """마지막 봉이 "close만 비어 있는" 상태인지 본다 (close 보완 대상 여부).

    입력: DataFrame(open, high, low, close, volume)
    출력: bool. close만 NaN이고 open/high/low가 모두 있으면 True.
    """
    if not len(df):
        return False
    last = df.iloc[-1]
    return bool(pd.isna(last["close"]) and last[["open", "high", "low"]].notna().all())


def _fill_last_close_from_meta(
    df: pd.DataFrame, meta: ChartMeta | None
) -> tuple[pd.DataFrame, bool, list[str]]:
    """마지막 봉의 빈 close를 chart API meta의 정규장 가격으로 채운다 (순수 함수).

    아래를 모두 만족할 때만 채운다 (P1.2 2번):
      1) 마지막 봉의 close만 NaN이고 open/high/low는 값이 있다
      2) meta.regularMarketTime이 그 봉과 같은 날짜이고, 정규장 마감(16:00 ET) 이후다
      3) meta.regularMarketPrice가 같은 날 low~high 범위 안이다
    하나라도 어긋나면 채우지 않는다. 채우지 않은 봉은 이후 "확정되지 않은 마지막 봉"
    제거 단계에서 버려진다.

    입력: DataFrame(open, high, low, close, volume, close_source), ChartMeta 또는 None
    출력: (보완된 DataFrame, 채웠는지 여부, 경고 메시지 목록)
    """
    if not _needs_close_fill(df):
        return df, False, []

    last_date = df.index[-1].date()
    date_str = last_date.isoformat()

    if meta is None or meta.price is None or meta.time_et is None:
        return df, False, [f"{date_str} close 비어 있음 - chart API meta를 쓸 수 없어 이 봉을 버림"]

    regular_close_et = datetime.combine(last_date, dtime(MARKET_CLOSE_HOUR, 0), tzinfo=US_EASTERN)
    if meta.time_et.date() != last_date or meta.time_et < regular_close_et:
        return df, False, [
            f"{date_str} close 비어 있음 - meta.regularMarketTime"
            f"({meta.time_et.isoformat()})이 이 날 정규장 마감 이후가 아니라 이 봉을 버림"
        ]

    low = float(df.iloc[-1]["low"])
    high = float(df.iloc[-1]["high"])
    price = float(meta.price)
    if not (low <= price <= high):
        return df, False, [
            f"{date_str} close 보완값 {price}이(가) 그날 저가~고가({low}~{high}) 범위를 벗어나 이 봉을 버림"
        ]

    out = df.copy()
    out.iloc[-1, out.columns.get_loc("close")] = price
    out.iloc[-1, out.columns.get_loc(_CLOSE_SOURCE_COLUMN)] = CLOSE_SOURCE_META
    return out, True, [f"{date_str} close를 chart API meta.regularMarketPrice({price})로 채움"]


def _clean_raw(
    raw: pd.DataFrame, now_et: datetime, meta_provider=None
) -> pd.DataFrame:
    """yfinance raw OHLCV를 정리한다.

    입력: yfinance Ticker.history(auto_adjust=False) 반환 DataFrame (tz-aware 인덱스),
         now_et(미국 동부 현재 시각),
         meta_provider(인자 없는 호출 가능 객체. close 보완이 필요할 때만 불러
                       ChartMeta 또는 None을 받는다. 없으면 보완하지 않는다)
    출력: DataFrame(open, high, low, close, volume, close_source), 날짜 오름차순, 확정 봉만.
         out.attrs["warnings"]에 실패는 아니지만 확인이 필요한 메시지를,
         out.attrs["close_filled_from_meta"]에 close 보완 여부를 담는다.
    """
    out = raw.rename(columns=_RAW_TO_LOWER)[_PRICE_COLUMNS].copy()
    out.index = out.index.tz_convert(US_EASTERN)
    out = out.sort_index()
    out[_CLOSE_SOURCE_COLUMN] = CLOSE_SOURCE_YAHOO

    if len(out) and now_et.hour < MARKET_CLOSE_HOUR and out.index[-1].date() == now_et.date():
        out = out.iloc[:-1]  # 미국장이 아직 안 끝난 당일 봉 제거

    warnings: list[str] = []

    # 야후가 최근 거래일의 close만 비워둔 채 내려주는 경우(P1.2 진단 결과)를
    # chart API meta의 정규장 종가로 채운다. 보완에 실패하면 채우지 않고,
    # 아래 "확정되지 않은 마지막 봉 제거"에서 그 봉이 버려진다.
    filled = False
    if _needs_close_fill(out) and meta_provider is not None:
        out, filled, fill_msgs = _fill_last_close_from_meta(out, meta_provider())
        warnings.extend(fill_msgs)

    # 마감 직후(또는 야후 데이터 파이프라인 지연)에는 종가 등 일부 값이 NaN인 채로
    # 내려오는 경우가 있다. open/high/low/close 중 하나라도 NaN이면 확정되지 않은
    # 값으로 보고 마지막 봉들을 제거한다. volume만 NaN이거나 0인 경우는 가격이
    # 이미 확정된 것이므로 버리지 않고 경고만 남긴다 (P1.1 1번).
    while len(out) and out.iloc[-1][_CORE_PRICE_COLUMNS].isna().any():
        out = out.iloc[:-1]

    if len(out):
        last_volume = out.iloc[-1]["volume"]
        if pd.isna(last_volume) or last_volume == 0:
            last_date = out.index[-1].date().isoformat()
            warnings.append(f"{last_date} 거래량이 NaN 또는 0 (봉은 유지, 확인 필요)")

    out.index = out.index.tz_localize(None).normalize()
    out.index.name = "date"
    out.attrs["warnings"] = warnings
    out.attrs["close_filled_from_meta"] = filled
    return out


def _fetch_chart_meta(ticker: str, timeout: float = 20.0) -> ChartMeta:
    """야후 chart API를 직접 호출해 meta의 정규장 가격·시각을 읽는다.

    yfinance는 quote 배열의 close만 쓰기 때문에, 야후가 close를 비워둔 날에는
    meta에 있는 정규장 종가를 가져올 수 없다. 그래서 직접 호출한다 (P1.2 2번).

    입력: yfinance 형식 ticker
    출력: ChartMeta(price, time_et)
    예외: 네트워크·응답 형식 오류는 그대로 올린다 (호출부에서 경고로 모은다)
    """
    resp = requests.get(
        CHART_API_URL.format(ticker=ticker),
        params={"range": "5d", "interval": "1d"},
        headers=_CHART_API_HEADERS,
        timeout=timeout,
    )
    resp.raise_for_status()
    meta = resp.json()["chart"]["result"][0]["meta"]

    price = meta.get("regularMarketPrice")
    ts = meta.get("regularMarketTime")
    time_et = (
        datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(US_EASTERN) if ts else None
    )
    return ChartMeta(price=price, time_et=time_et)


def _restore_close_from_cache(fresh: pd.DataFrame, cached: pd.DataFrame | None) -> tuple[pd.DataFrame, int]:
    """새로 받은 데이터 중간에 빈 close를, 예전에 meta로 채워 캐시해 둔 값으로 되살린다.

    meta.regularMarketPrice는 "가장 최근" 정규장 값 하나뿐이라, 야후가 close를
    비워둔 날이 더 이상 마지막 봉이 아니게 되면 그 날은 다시 채울 수 없다. 같은
    조건(그날 저가~고가 범위)을 다시 확인한 뒤 캐시에 있던 값을 쓴다.

    입력: 새로 정리한 DataFrame, 캐시 DataFrame(없으면 None)
    출력: (되살린 DataFrame, 되살린 봉 수)
    """
    if cached is None or cached.empty or _CLOSE_SOURCE_COLUMN not in cached.columns:
        return fresh, 0

    missing = fresh.index[fresh["close"].isna()]
    donors = cached[cached[_CLOSE_SOURCE_COLUMN] == CLOSE_SOURCE_META]
    dates = missing.intersection(donors.index)
    if not len(dates):
        return fresh, 0

    out = fresh.copy()
    restored = 0
    for d in dates:
        price = float(donors.loc[d, "close"])
        low, high = out.loc[d, "low"], out.loc[d, "high"]
        if pd.isna(low) or pd.isna(high) or not (float(low) <= price <= float(high)):
            continue
        out.loc[d, "close"] = price
        out.loc[d, _CLOSE_SOURCE_COLUMN] = CLOSE_SOURCE_META
        restored += 1
    return out, restored


def find_mid_series_gaps(df: pd.DataFrame) -> list:
    """마지막 봉이 아닌 곳에 close가 NaN인 날짜를 찾는다 (P2 P1 마무리 2번).

    중간에 close 구멍이 있으면 조용히 보간하거나 채우지 않고, 그 날짜를 그대로
    보고한다 — 호출부가 data_gap 표시와 신호 판정 제외에 쓴다.

    입력: DataFrame(..., close)
    출력: 구멍이 있는 날짜(index 값) 목록, 오름차순. 없으면 빈 리스트.
    """
    if len(df) < 2:
        return []
    mid = df.iloc[:-1]
    return list(mid.index[mid["close"].isna()])


def has_meta_close(df: pd.DataFrame) -> bool:
    """마지막 봉의 close가 meta 보완값인지 본다.

    입력: DataFrame(..., close_source)
    출력: bool. 캐시에서 그대로 읽어온 경우에도 데이터만 보고 판정할 수 있다.
    """
    if not len(df) or _CLOSE_SOURCE_COLUMN not in df.columns:
        return False
    return df.iloc[-1][_CLOSE_SOURCE_COLUMN] == CLOSE_SOURCE_META


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
        if _CLOSE_SOURCE_COLUMN not in df.columns:  # close_source 도입(P1.2) 이전 캐시
            df[_CLOSE_SOURCE_COLUMN] = CLOSE_SOURCE_YAHOO
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

    마지막 봉의 close가 chart API meta 보완값(close_source="meta")이면 신선하다고
    보지 않는다. 야후가 그 사이 실제 일봉 종가를 채워 넣었을 수 있어, 매번 다시
    받아서 확인해야 한다 (P2 P1 마무리 1번 — 일봉 종가가 meta 보완값보다 우선).
    """
    if cached is None or cached.empty:
        return False
    latest_needed = _latest_confirmed_trading_date(now_et)
    if cached.index[-1].date() < latest_needed:
        return False
    return not has_meta_close(cached)


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
        cached.attrs.setdefault("close_filled_from_meta", False)
        # parquet은 DataFrame.attrs를 보존하지 않으므로 캐시를 그대로 돌려줄 때도
        # data_gap 여부는 매번 다시 계산한다.
        gap_dates = find_mid_series_gaps(cached)
        if gap_dates:
            dates_str = ", ".join(d.date().isoformat() for d in gap_dates)
            cached.attrs["warnings"].append(
                f"[data_gap] 중간에 close가 비어 있는 봉 {len(gap_dates)}개 ({dates_str}) - "
                "보간하지 않음, 해당 날짜는 신호 판정에서 제외해야 함"
            )
        cached.attrs["data_gap_dates"] = gap_dates
        return cached

    raw = yf.Ticker(ticker).history(period=_period_for(history_days), auto_adjust=False)
    if raw.empty:
        raise ValueError("가격 데이터 없음")

    # close 보완이 필요할 때만 chart API를 한 번 더 부른다. 네트워크 실패는
    # 조용히 넘기지 않고 경고로 모은다 (보완은 못 하고 그 봉은 버려진다).
    meta_errors: list[str] = []

    def _meta_provider() -> ChartMeta | None:
        try:
            return _fetch_chart_meta(ticker)
        except Exception as exc:
            meta_errors.append(f"chart API meta 조회 실패: {exc}")
            return None

    cleaned = _clean_raw(raw, now_et, meta_provider=_meta_provider)
    cleaned, restored = _restore_close_from_cache(cleaned, cached)
    if cleaned.empty:
        raise ValueError("확정된 가격 데이터 없음 (전부 NaN)")

    warnings = cleaned.attrs.get("warnings", []) + meta_errors
    if restored:
        warnings.append(f"중간에 비어 있던 close {restored}개를 캐시의 meta 보완값으로 되살림")

    still_missing = cleaned.index[cleaned["close"].isna()]
    if len(still_missing):
        dates = ", ".join(d.date().isoformat() for d in still_missing[-5:])
        warnings.append(f"close가 비어 있는 봉 {len(still_missing)}개가 남아 있음 (최근: {dates})")

    # 마지막 봉이 아닌 곳의 close 구멍은 조용히 채우지 않고 크게 경고한다
    # (P2 P1 마무리 2번). 신호 판정은 engine에서 이 날짜를 제외한다.
    gap_dates = find_mid_series_gaps(cleaned)
    if gap_dates:
        dates_str = ", ".join(d.date().isoformat() for d in gap_dates)
        warnings.append(
            f"[data_gap] 중간에 close가 비어 있는 봉 {len(gap_dates)}개 ({dates_str}) - "
            "보간하지 않음, 해당 날짜는 신호 판정에서 제외해야 함"
        )
    cleaned.attrs["data_gap_dates"] = gap_dates

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
    cleaned.attrs["close_filled_from_meta"] = bool(
        cleaned.attrs.get("close_filled_from_meta") or restored
    )
    _save_cache(ticker, cleaned)
    return cleaned


def fetch_universe_prices(tickers: list[str], cfg: dict, sleep_sec: float = 0.3) -> PriceFetchResult:
    """여러 종목의 확정 일봉을 받는다. 실패 종목은 모아서 반환하고 계속 진행한다.

    입력: yfinance 형식 ticker 목록, cfg(config.yaml 로드값)
    출력: PriceFetchResult(prices, failed, warnings, close_filled)
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
            if has_meta_close(df):  # 캐시 재사용 시에도 데이터만 보고 센다
                result.close_filled.append(ticker)
            gap_dates = df.attrs.get("data_gap_dates") or []
            if gap_dates:
                result.data_gap[ticker] = [d.date().isoformat() for d in gap_dates]
        except Exception as exc:
            result.failed[ticker] = str(exc)
        time.sleep(sleep_sec)

    print(f"[prices] close 보완(meta) 종목 수: {len(result.close_filled)}")
    if result.close_filled:
        print("[prices] close 보완 티커:", ", ".join(result.close_filled))
    if result.data_gap:
        print(f"[prices] *** 경고: data_gap 종목 {len(result.data_gap)}개 (중간 close 구멍, 보간하지 않음) ***")
        for ticker, dates in result.data_gap.items():
            print(f"  - {ticker}: {', '.join(dates)}")
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
