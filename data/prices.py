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
- close를 못 구한 봉은 meta → 캐시(예전에 meta로 채워 둔 값) → 예비 출처(60분봉
  마지막 봉 종가) 순으로 채워본다 (P3.2 1번). 이 순서를 지키지 않으면(캐시 확인
  전에 봉을 버리면) 캐시에 있던 값이 있어도 기준일이 뒤로 가는 회귀가 생긴다.
- 행이 통째로 사라지는 문제(P3.3): "확정되지 않은 마지막 봉 제거" 단계(마지막
  봉의 open/high/low/close 중 하나라도 NaN이면 그 봉을 지운다)를 통과한 뒤
  `_save_cache`가 정리된 DataFrame을 파일 전체를 덮어쓰는 방식으로 저장한다.
  두 단계가 겹치면: 야후가 어떤 날 하루만 close를 비워 내려주고(또는 그날 자체를
  응답에서 빠뜨리고) meta·예비 출처도 실패하면, 그 봉이 트림으로 지워진 채
  캐시 전체가 그대로 덮어써져 예전엔 확정돼 있던 봉이 영구히 사라진다(9/22
  CSX·ODFL·PEP 사례의 원인). `_restore_missing_rows_from_cache`가 트림 직후
  캐시에 남아 있던 확정 봉(close가 NaN이 아니었던 날짜)을 되살려 이 경로를 막는다.
- 그래도 캐시에 없던 날짜(새 구멍)는 `find_mid_series_gaps`가 NYSE 거래일 달력
  기준으로 찾아내고(NaN 행 + 아예 없는 행 모두), `recover_gap_days`가 그 날짜만
  다시 받아 채운다(일봉 재조회 → 60분봉 집계 → 실패 시 data_gap, P3.3 2번).
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

from data import market_calendar  # noqa: E402
from data.market_calendar import latest_closed_trading_day  # noqa: E402

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
# "meta" = chart API의 meta.regularMarketPrice로 채운 값,
# "fallback_60m" = 예비 출처(60분봉 마지막 봉 종가)로 채운 값 (P3.2 1번).
_CLOSE_SOURCE_COLUMN = "close_source"
CLOSE_SOURCE_YAHOO = "yahoo"
CLOSE_SOURCE_META = "meta"
CLOSE_SOURCE_FALLBACK = "fallback_60m"
# "hourly" = 누락 거래일 복구(P3.3)에서 60분봉을 집계해 하루치 봉 전체(시가·고가·
# 저가·거래량까지)를 새로 만든 값. close만 채우는 CLOSE_SOURCE_FALLBACK과 달리
# 행 자체가 캐시에 아예 없던 경우에 쓴다.
CLOSE_SOURCE_HOURLY = "hourly"

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
    출력: date. NYSE 거래일 달력(주말·공휴일·조기 폐장 반영, P3.2 2번)으로
         가장 최근에 마감된 거래일을 구한다.
    """
    return latest_closed_trading_day(now_et)


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


# meta.regularMarketTime을 받아들이는 허용 폭 (P2.1 보완 3번). 정규장 마감(16:00 ET)
# 이후 이 폭을 넘겨 받은 시각이면 장후 거래 등 살아있는(계속 바뀌는) 값일 수 있어
# 보완에 쓰지 않는다 — 그래야 같은 날 여러 번 실행해도 같은 값이 나온다.
_META_CLOSE_TOLERANCE = timedelta(minutes=5)


def _fill_last_close_from_meta(
    df: pd.DataFrame, meta: ChartMeta | None
) -> tuple[pd.DataFrame, bool, list[str], str | None]:
    """마지막 봉의 빈 close를 chart API meta의 정규장 가격으로 채운다 (순수 함수).

    아래를 모두 만족할 때만 채운다 (P1.2 2번, P2.1 보완 3번):
      1) 마지막 봉의 close만 NaN이고 open/high/low는 값이 있다
      2) meta.regularMarketTime이 그 봉과 같은 날짜이고, 정규장 마감(16:00 ET)부터
         _META_CLOSE_TOLERANCE 이내다 — 그보다 이르면 장중, 늦으면 장후(살아있는) 값이다.
      3) meta.regularMarketPrice가 같은 날 low~high 범위 안이다
    하나라도 어긋나면 채우지 않는다. 채우지 않은 봉은 이후 "확정되지 않은 마지막 봉"
    제거 단계에서 버려진다.

    입력: DataFrame(open, high, low, close, volume, close_source), ChartMeta 또는 None
    출력: (보완된 DataFrame, 채웠는지 여부, 경고 메시지 목록, 사용한 meta.regularMarketTime
          ISO 문자열(채우지 못했으면 None) — 실행 로그·재현성 확인용)
    """
    if not _needs_close_fill(df):
        return df, False, [], None

    last_date = df.index[-1].date()
    date_str = last_date.isoformat()

    if meta is None or meta.price is None or meta.time_et is None:
        return df, False, [f"{date_str} close 비어 있음 - chart API meta를 쓸 수 없어 이 봉을 버림"], None

    regular_close_et = datetime.combine(last_date, dtime(MARKET_CLOSE_HOUR, 0), tzinfo=US_EASTERN)
    window_end = regular_close_et + _META_CLOSE_TOLERANCE
    if meta.time_et.date() != last_date or not (regular_close_et <= meta.time_et <= window_end):
        when = "이전(장중)" if meta.time_et < regular_close_et else "이후(장후·라이브 가격일 수 있음)"
        return df, False, [
            f"{date_str} close 비어 있음 - meta.regularMarketTime"
            f"({meta.time_et.isoformat()})이 이 날 정규장 마감 허용 폭({when})을 벗어나 이 봉을 버림"
        ], None

    low = float(df.iloc[-1]["low"])
    high = float(df.iloc[-1]["high"])
    price = float(meta.price)
    if not (low <= price <= high):
        return df, False, [
            f"{date_str} close 보완값 {price}이(가) 그날 저가~고가({low}~{high}) 범위를 벗어나 이 봉을 버림"
        ], None

    out = df.copy()
    out.iloc[-1, out.columns.get_loc("close")] = price
    out.iloc[-1, out.columns.get_loc(_CLOSE_SOURCE_COLUMN)] = CLOSE_SOURCE_META
    return out, True, [f"{date_str} close를 chart API meta.regularMarketPrice({price})로 채움"], meta.time_et.isoformat()


def _clean_raw(
    raw: pd.DataFrame, now_et: datetime, meta_provider=None, cache_df: pd.DataFrame | None = None,
    fallback_provider=None,
) -> pd.DataFrame:
    """yfinance raw OHLCV를 정리한다.

    입력: yfinance Ticker.history(auto_adjust=False) 반환 DataFrame (tz-aware 인덱스),
         now_et(미국 동부 현재 시각),
         meta_provider(인자 없는 호출 가능 객체. close 보완이 필요할 때만 불러
                       ChartMeta 또는 None을 받는다. 없으면 보완하지 않는다),
         cache_df(예전에 캐시해 둔 DataFrame. close 보완용 — 없으면 None),
         fallback_provider(date -> float|None 호출 가능 객체. meta·캐시로도 못
                            채운 close에 예비 출처를 쓴다 — 없으면 시도하지 않는다)
    출력: DataFrame(open, high, low, close, volume, close_source), 날짜 오름차순, 확정 봉만.
         out.attrs["warnings"]에 실패는 아니지만 확인이 필요한 메시지를,
         out.attrs["close_filled_from_meta"]에 close 보완(meta 또는 캐시) 여부를 담는다.

    close 보완 순서(P3.2 1번, 중요): meta → 캐시 → 예비 출처. 마지막 봉이 아직
    확정 안 된 값으로 보고 버려지는 "확정되지 않은 마지막 봉 제거" 단계보다
    반드시 먼저 시도해야 한다 — 그 단계 뒤에 시도하면 이미 버려진 봉이라
    캐시에 값이 있어도 되살릴 수 없다(P3.1 실행에서 실제로 난 회귀 원인).
    """
    out = raw.rename(columns=_RAW_TO_LOWER)[_PRICE_COLUMNS].copy()
    out.index = out.index.tz_convert(US_EASTERN)
    out = out.sort_index()
    out[_CLOSE_SOURCE_COLUMN] = CLOSE_SOURCE_YAHOO

    if len(out) and now_et.hour < MARKET_CLOSE_HOUR and out.index[-1].date() == now_et.date():
        out = out.iloc[:-1]  # 미국장이 아직 안 끝난 당일 봉 제거

    # cache_df·find_mid_series_gaps 등 나머지 로직이 모두 이 tz-naive 정규화된
    # 인덱스를 기준으로 하므로, close 보완 단계 전에 먼저 맞춰 둔다.
    out.index = out.index.tz_localize(None).normalize()
    out.index.name = "date"

    warnings: list[str] = []

    # 야후가 최근 거래일의 close만 비워둔 채 내려주는 경우(P1.2 진단 결과)를
    # chart API meta의 정규장 종가로 채운다.
    filled = False
    meta_time_used: str | None = None
    if _needs_close_fill(out) and meta_provider is not None:
        out, filled, fill_msgs, meta_time_used = _fill_last_close_from_meta(out, meta_provider())
        warnings.extend(fill_msgs)

    # meta로 못 채웠으면 캐시(예전에 meta로 채워 둔 값)로, 그래도 못 채웠으면
    # 예비 출처(60분봉)로 채워본다. 셋 다 실패한 채 아래 "확정되지 않은 마지막
    # 봉 제거"에 걸리면 그 봉이 버려진다(다음 실행에서 다시 시도).
    cache_restored = 0
    if cache_df is not None:
        out, cache_restored = _restore_close_from_cache(out, cache_df)
        if cache_restored:
            warnings.append(f"close 비어 있는 봉 {cache_restored}개를 캐시의 meta 보완값으로 되살림")

    fallback_restored = 0
    if fallback_provider is not None:
        out, fallback_restored, fb_msgs = _restore_close_from_fallback(out, fallback_provider)
        warnings.extend(fb_msgs)

    # 마감 직후(또는 야후 데이터 파이프라인 지연)에는 종가 등 일부 값이 NaN인 채로
    # 내려오는 경우가 있다. open/high/low/close 중 하나라도 NaN이면 확정되지 않은
    # 값으로 보고 마지막 봉들을 제거한다. volume만 NaN이거나 0인 경우는 가격이
    # 이미 확정된 것이므로 버리지 않고 경고만 남긴다 (P1.1 1번).
    while len(out) and out.iloc[-1][_CORE_PRICE_COLUMNS].isna().any():
        out = out.iloc[:-1]

    # 트림으로 지워졌거나 이번 응답 자체에 없던 날짜 중, 예전 캐시에 확정 봉으로
    # 남아 있던 날짜는 되살린다 — 새로 받은 데이터가 비어 있다는 이유로 이미
    # 확정된 과거 봉이 사라지지 않게 막는다 (P3.3 2번).
    out, rows_restored = _restore_missing_rows_from_cache(out, cache_df)
    if rows_restored:
        warnings.append(f"이전에 확정됐던 행 {rows_restored}개가 이번 응답에는 없어 캐시에서 되살림")

    if len(out):
        last_volume = out.iloc[-1]["volume"]
        if pd.isna(last_volume) or last_volume == 0:
            last_date = out.index[-1].date().isoformat()
            warnings.append(f"{last_date} 거래량이 NaN 또는 0 (봉은 유지, 확인 필요)")

    out.attrs["warnings"] = warnings
    out.attrs["close_filled_from_meta"] = bool(filled or cache_restored or fallback_restored)
    out.attrs["close_meta_time"] = meta_time_used
    out.attrs["close_cache_restored"] = cache_restored
    out.attrs["close_fallback_restored"] = fallback_restored
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


def _fetch_intraday_last_close(ticker: str, target_date: date) -> float | None:
    """그 날짜의 60분봉 마지막 봉 종가를 예비 종가 출처로 가져온다 (P3.2 1번).

    meta도 캐시도 못 쓸 때만 부른다. Stooq 일봉 CSV는 이 환경에서 봇 차단(JS
    챌린지, HTTP 200으로 위장한 JS 페이지)에 막혀 직접 확인 결과 쓸 수 없었다
    (완료 보고에 기록) — 그래서 야후 60분봉만 쓴다.

    입력: yfinance 형식 ticker, target_date(그날 종가가 필요한 날짜)
    출력: 그날 마지막 60분봉의 종가, 없으면 None
    예외: 네트워크·응답 형식 오류는 그대로 올린다 (호출부에서 경고로 모은다)
    """
    raw = yf.Ticker(ticker).history(period="7d", interval="60m", auto_adjust=False)
    if raw.empty:
        return None
    idx = raw.index.tz_convert(US_EASTERN)
    day_bars = raw.loc[idx.date == target_date]
    if day_bars.empty:
        return None
    return float(day_bars.iloc[-1]["Close"])


def _restore_close_from_fallback(df: pd.DataFrame, fallback_provider) -> tuple[pd.DataFrame, int, list[str]]:
    """meta·캐시로도 못 채운 close를 예비 출처로 채운다 (P3.2 1번, 순수 함수).

    입력: DataFrame(..., low, high, close, close_source), fallback_provider
         (date -> float 또는 None을 돌려주는 호출 가능 객체. 네트워크 호출은
         호출부가 감싸 예외를 경고로 모은다)
    출력: (채운 DataFrame, 채운 봉 수, 경고 메시지 목록). 그날 저가~고가 범위 밖
         값은 채우지 않고 경고만 남긴다.
    """
    missing = df.index[df["close"].isna()]
    if not len(missing):
        return df, 0, []

    out = df.copy()
    filled = 0
    warnings: list[str] = []
    for d in missing:
        low, high = out.loc[d, "low"], out.loc[d, "high"]
        if pd.isna(low) or pd.isna(high):
            continue
        price = fallback_provider(d.date())
        if price is None:
            continue
        if not (float(low) <= price <= float(high)):
            warnings.append(
                f"{d.date()} 예비 종가(60분봉) {price}이(가) 그날 저가~고가({low}~{high}) 범위를 벗어나 버림"
            )
            continue
        out.loc[d, "close"] = price
        out.loc[d, _CLOSE_SOURCE_COLUMN] = CLOSE_SOURCE_FALLBACK
        warnings.append(f"{d.date()} close를 예비 출처(60분봉 마지막 봉)로 채움({price})")
        filled += 1
    return out, filled, warnings


def _restore_close_from_cache(fresh: pd.DataFrame, cached: pd.DataFrame | None) -> tuple[pd.DataFrame, int]:
    """새로 받은 데이터 중간에 빈 close를, 예전에 캐시해 둔 확정 값으로 되살린다.

    meta.regularMarketPrice는 "가장 최근" 정규장 값 하나뿐이라, 야후가 close를
    비워둔 날이 더 이상 마지막 봉이 아니게 되면 그 날은 다시 채울 수 없다. 같은
    조건(그날 저가~고가 범위)을 다시 확인한 뒤 캐시에 있던 값을 쓴다.

    캐시에 close가 있는 날짜(출처가 meta든 yahoo든 hourly든)는 모두 "확정된 과거
    봉"이라 되살리기 후보다(P3.3 2번 — 새로 받은 응답이 비어 있다는 이유로 이미
    확정된 값을 잃지 않게 한다). 출처를 meta로만 좁히지 않는다.

    입력: 새로 정리한 DataFrame, 캐시 DataFrame(없으면 None)
    출력: (되살린 DataFrame, 되살린 봉 수)
    """
    if cached is None or cached.empty or _CLOSE_SOURCE_COLUMN not in cached.columns:
        return fresh, 0

    missing = fresh.index[fresh["close"].isna()]
    donors = cached[cached["close"].notna()]
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
        out.loc[d, _CLOSE_SOURCE_COLUMN] = donors.loc[d, _CLOSE_SOURCE_COLUMN]
        restored += 1
    return out, restored


def _restore_missing_rows_from_cache(fresh: pd.DataFrame, cached: pd.DataFrame | None) -> tuple[pd.DataFrame, int]:
    """새 응답에 아예 없는 날짜를, 예전 캐시에 확정 봉으로 남아 있던 값으로 되살린다.

    "확정되지 않은 마지막 봉 제거" 단계에서 트림된 날짜나, 이번 야후 응답 자체가
    건너뛴 날짜가 대상이다. 캐시에 close가 있던(=확정됐던) 날짜인데 새 데이터의
    인덱스에 통째로 없으면, open/high/low/close/volume/close_source를 캐시 값
    그대로 되살린다 (P3.3 2번 — 9/22 CSX·ODFL·PEP 소실 재발 방지).

    입력: fresh(트림까지 끝난 정리 중인 DataFrame), cached(예전 캐시, 없으면 None)
    출력: (되살린 DataFrame, 되살린 행 수)
    """
    if cached is None or cached.empty or _CLOSE_SOURCE_COLUMN not in cached.columns:
        return fresh, 0

    confirmed = cached[cached["close"].notna()]
    missing_dates = confirmed.index.difference(fresh.index)
    if not len(missing_dates):
        return fresh, 0

    cols = [c for c in _PRICE_COLUMNS if c in fresh.columns] + [_CLOSE_SOURCE_COLUMN]
    restored_rows = confirmed.loc[missing_dates, cols]
    out = pd.concat([fresh, restored_rows]).sort_index()
    return out, len(missing_dates)


def find_mid_series_gaps(df: pd.DataFrame) -> list:
    """마지막 봉이 아닌 곳에서 close가 NaN이거나 거래일 자체가 인덱스에 아예 없는
    날짜를 찾는다 (P2 P1 마무리 2번, P3.3 1번 — 통째로 사라진 행도 구멍으로 잡는다).

    NaN 행은 기존처럼 df 안에서 바로 찾고, 통째로 없는 행은 df가 가진 기간(첫
    날짜~마지막 날짜) 안의 NYSE 거래일 달력(data/market_calendar.py)과 df 인덱스의
    차집합으로 찾는다. 상장 전 기간은 df 자체에 없으므로 자동으로 탐지 범위 밖이다.
    중간에 구멍이 있으면 조용히 보간하거나 채우지 않고, 그 날짜를 그대로 보고한다
    — 호출부가 data_gap 표시와 신호 판정 제외, 복구(recover_gap_days)에 쓴다.

    입력: DataFrame(..., close), 날짜 오름차순 인덱스
    출력: 구멍이 있는 날짜(index 값) 목록, 오름차순. 없으면 빈 리스트.
    """
    if len(df) < 2:
        return []
    mid = df.iloc[:-1]
    nan_gaps = set(mid.index[mid["close"].isna()])

    last_date = df.index[-1]
    expected = market_calendar.trading_days_between(df.index[0].date(), last_date.date())
    missing_rows = {d for d in expected if d < last_date} - set(df.index)

    return sorted(nan_gaps | missing_rows)


def recover_gap_days(
    df: pd.DataFrame, gap_dates: list, daily_provider, hourly_provider
) -> tuple[pd.DataFrame, list, list, list]:
    """find_mid_series_gaps가 찾은 구멍 날짜를 하루씩 다시 받아 채운다 (순수 함수, P3.3 2번).

    우선순위: 일봉 재조회 종가(daily_provider) → 60분봉 집계(hourly_provider, 시가·
    고가·저가·거래량도 60분봉에서 새로 만든다) → 실패. 두 출처 모두 그날
    저가~고가 범위를 벗어나는 값은 쓰지 않고 실패로 취급한다.

    입력: df(open,high,low,close,volume,close_source), gap_dates(find_mid_series_gaps
         결과), daily_provider(date -> {open,high,low,close,volume} 또는 None. 예외를
         던지면 실패로 보고 다음 출처로 넘어간다), hourly_provider(위와 같은 시그니처)
    출력: (복구된 df, 복구한 날짜 목록, 끝내 복구 못 한 날짜 목록, 경고 메시지 목록)
    """
    out = df.copy()
    recovered: list = []
    unresolved: list = []
    warnings: list[str] = []

    for d in gap_dates:
        target = d.date() if hasattr(d, "date") else d
        values, source = None, None

        try:
            values = daily_provider(target)
            source = CLOSE_SOURCE_YAHOO
        except Exception as exc:
            warnings.append(f"{target} 누락 거래일 일봉 재조회 실패: {exc}")

        if values is None:
            try:
                values = hourly_provider(target)
                source = CLOSE_SOURCE_HOURLY
            except Exception as exc:
                warnings.append(f"{target} 누락 거래일 60분봉 조회 실패: {exc}")

        if values is None:
            unresolved.append(d)
            warnings.append(f"{target} 누락 거래일 복구 실패 (일봉·60분봉 모두 실패) - data_gap 처리")
            continue

        low, high, close = values.get("low"), values.get("high"), values.get("close")
        if low is None or high is None or close is None or not (float(low) <= float(close) <= float(high)):
            unresolved.append(d)
            warnings.append(f"{target} 복구값이 그날 저가~고가 범위를 벗어나 버림 - data_gap 처리")
            continue

        for col in ("open", "high", "low", "close", "volume"):
            out.loc[d, col] = values[col]
        out.loc[d, _CLOSE_SOURCE_COLUMN] = source
        recovered.append(d)
        warnings.append(f"{target} 누락 거래일 복구함 (close_source={source})")

    out = out.sort_index()
    return out, recovered, unresolved, warnings


def _fetch_daily_bar(ticker: str, target_date: date) -> dict | None:
    """그 날짜 하루치 일봉을 다시 받는다 (누락 거래일 복구 1순위, P3.3 2번).

    입력: yfinance 형식 ticker, target_date
    출력: {open,high,low,close,volume} 또는 None(그 날짜 봉이 없거나 close가 비어 있음)
    예외: 네트워크·응답 형식 오류는 그대로 올린다 (호출부가 경고로 모으고 다음
         출처로 넘어간다)
    """
    raw = yf.Ticker(ticker).history(
        start=target_date, end=target_date + timedelta(days=1), auto_adjust=False
    )
    if raw.empty:
        return None
    row = raw.iloc[0]
    close = row.get("Close")
    if close is None or pd.isna(close):
        return None
    return {
        "open": float(row["Open"]),
        "high": float(row["High"]),
        "low": float(row["Low"]),
        "close": float(close),
        "volume": float(row.get("Volume") or 0),
    }


def _fetch_hourly_bar(ticker: str, target_date: date) -> dict | None:
    """그 날짜의 60분봉을 모아 하루치 봉으로 집계한다 (누락 거래일 복구 2순위, P3.3 2번).

    시가=그날 첫 60분봉의 시가, 고가·저가=그날 60분봉 중 최댓값·최솟값, 종가=그날
    마지막 60분봉의 종가, 거래량=그날 60분봉 거래량 합.

    입력: yfinance 형식 ticker, target_date
    출력: {open,high,low,close,volume} 또는 None(그날 60분봉이 없음)
    예외: 네트워크·응답 형식 오류는 그대로 올린다 (호출부에서 경고로 모은다)
    """
    raw = yf.Ticker(ticker).history(
        start=target_date, end=target_date + timedelta(days=1), interval="60m", auto_adjust=False
    )
    if raw.empty:
        return None
    idx = raw.index.tz_convert(US_EASTERN)
    day_bars = raw.loc[idx.date == target_date]
    if day_bars.empty:
        return None
    return {
        "open": float(day_bars.iloc[0]["Open"]),
        "high": float(day_bars["High"].max()),
        "low": float(day_bars["Low"].min()),
        "close": float(day_bars.iloc[-1]["Close"]),
        "volume": float(day_bars["Volume"].sum()),
    }


def _daily_bar_provider(ticker: str):
    return lambda target_date: _fetch_daily_bar(ticker, target_date)


def _hourly_bar_provider(ticker: str):
    return lambda target_date: _fetch_hourly_bar(ticker, target_date)


def _expand_with_next_trading_day(index: pd.DatetimeIndex, gap_dates: list) -> list:
    """끝내 복구 못 한 구멍 날짜 바로 다음 날짜도 data_gap에 넣는다 (P3.3 3번).

    core.state.check_a1 등은 df.iloc[idx-1]로 "전날"을 찾는다. 구멍이 그대로
    남으면 구멍 다음 날짜의 "전날"이 실제 직전 거래일이 아니라 그 전전날이
    되어 신호가 잘못될 수 있다(9/22가 빈 채로 9/21→9/23을 비교해 A1이 9/23으로
    밀린 사례). 구멍 날짜 자체는 이미 data_gap이라 그날 판정에서 빠지지만,
    "전날"이 어긋나는 것은 구멍의 다음 거래일이므로 그 날짜도 함께 넣어야
    core.state가 그 어긋난 비교로 신호를 내지 않는다(신호 없음 + data_gap 경고).

    입력: DataFrame 인덱스(오름차순), gap_dates(끝내 복구 못 한 날짜 목록)
    출력: gap_dates + 그 다음 실제 존재하는 날짜, 오름차순, 중복 없음
    """
    if not len(gap_dates):
        return list(gap_dates)
    expanded = set(gap_dates)
    for d in gap_dates:
        later = index[index > d]
        if len(later):
            expanded.add(later[0])
    return sorted(expanded)


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
        cached.attrs.setdefault("close_meta_time", None)  # 이번 실행에서 새로 채우지 않았다
        # parquet은 DataFrame.attrs를 보존하지 않으므로 캐시를 그대로 돌려줄 때도
        # data_gap 여부는 매번 다시 계산한다. 이전 실행에서 남은 구멍도 여기서
        # 다시 복구를 시도한다(P3.3 1·2번).
        gap_dates = find_mid_series_gaps(cached)
        if gap_dates:
            cached, recovered_dates, unresolved_dates, recovery_warnings = recover_gap_days(
                cached, gap_dates, _daily_bar_provider(ticker), _hourly_bar_provider(ticker)
            )
            cached.attrs["warnings"].extend(recovery_warnings)
            if recovered_dates:
                _save_cache(ticker, cached)
            if unresolved_dates:
                dates_str = ", ".join(str(d.date()) for d in unresolved_dates)
                cached.attrs["warnings"].append(
                    f"[data_gap] 끝내 복구 못 한 거래일 {len(unresolved_dates)}개 ({dates_str}) - "
                    "신호 판정에서 제외해야 함"
                )
            gap_dates = _expand_with_next_trading_day(cached.index, unresolved_dates)
        cached.attrs["data_gap_dates"] = gap_dates
        return cached

    raw = yf.Ticker(ticker).history(period=_period_for(history_days), auto_adjust=False)
    if raw.empty:
        raise ValueError("가격 데이터 없음")

    # close 보완이 필요할 때만 chart API·60분봉을 추가로 부른다. 네트워크 실패는
    # 조용히 넘기지 않고 경고로 모은다 (그 출처는 못 쓰고 다음 출처로 넘어간다).
    meta_errors: list[str] = []

    def _meta_provider() -> ChartMeta | None:
        try:
            return _fetch_chart_meta(ticker)
        except Exception as exc:
            meta_errors.append(f"chart API meta 조회 실패: {exc}")
            return None

    fallback_errors: list[str] = []

    def _fallback_provider(target_date: date) -> float | None:
        try:
            return _fetch_intraday_last_close(ticker, target_date)
        except Exception as exc:
            fallback_errors.append(f"{target_date} 예비 종가(60분봉) 조회 실패: {exc}")
            return None

    # close 보완 순서: meta(당일 실시간) -> 캐시(예전 meta 보완값) -> 예비 출처(60분봉).
    # 캐시 확인은 "확정되지 않은 마지막 봉 제거"보다 먼저 해야 한다 — 그래야
    # 캐시에 값이 있는데도 봉이 통째로 버려져 기준일이 뒤로 가는 일이 없다.
    cleaned = _clean_raw(
        raw, now_et, meta_provider=_meta_provider, cache_df=cached, fallback_provider=_fallback_provider
    )
    if cleaned.empty:
        raise ValueError("확정된 가격 데이터 없음 (전부 NaN)")

    warnings = cleaned.attrs.get("warnings", []) + meta_errors + fallback_errors

    still_missing = cleaned.index[cleaned["close"].isna()]
    if len(still_missing):
        dates = ", ".join(d.date().isoformat() for d in still_missing[-5:])
        warnings.append(f"close가 비어 있는 봉 {len(still_missing)}개가 남아 있음 (최근: {dates})")

    # 마지막 봉이 아닌 곳의 구멍(NaN close, 또는 행 자체가 없음)은 그 날짜만 다시
    # 받아 복구를 시도한다 (P3.3 1·2번). 끝내 복구 못 한 날짜만 data_gap으로
    # 남기고 조용히 보간하지 않는다. 신호 판정은 engine에서 이 날짜를 제외한다.
    gap_dates = find_mid_series_gaps(cleaned)
    if gap_dates:
        cleaned, recovered_dates, unresolved_dates, recovery_warnings = recover_gap_days(
            cleaned, gap_dates, _daily_bar_provider(ticker), _hourly_bar_provider(ticker)
        )
        warnings.extend(recovery_warnings)
        if unresolved_dates:
            dates_str = ", ".join(str(d.date()) for d in unresolved_dates)
            warnings.append(
                f"[data_gap] 끝내 복구 못 한 거래일 {len(unresolved_dates)}개 ({dates_str}) - "
                "신호 판정에서 제외해야 함"
            )
        gap_dates = _expand_with_next_trading_day(cleaned.index, unresolved_dates)
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
