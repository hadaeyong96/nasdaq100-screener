"""원/달러 환율(yfinance `KRW=X`)을 받아 캐시한다 (P3.6 6-4번).

기준일 종가 환율을 쓴다. `data/cache/fx_krw.json`에 {날짜: 환율} 이력을 캐시해
두고, 그날 값을 새로 못 받으면(네트워크 실패 등) 캐시에 남아 있는 가장 최근
값을 대신 쓰고 경고를 남긴다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
CACHE_PATH = DATA_DIR / "cache" / "fx_krw.json"


@dataclass
class FxRateResult:
    """환율 조회 결과.

    rate: 원/달러 환율 (구하지 못했으면 None)
    rate_date: 실제로 쓴 환율의 날짜(YYYY-MM-DD) — 기준일과 다르면 캐시 대체값
    is_fallback: 기준일 값을 새로 못 받아 캐시의 예전 값을 대신 썼으면 True
    warning: 확인이 필요한 메시지 (없으면 None)
    """

    rate: float | None
    rate_date: str | None
    is_fallback: bool
    warning: str | None = None


def _load_cache() -> dict[str, float]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(history: dict[str, float]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")


def pick_rate_for_date(history: dict[str, float], target_date: str) -> tuple[float, str] | None:
    """history({날짜: 환율}) 중 target_date 이하에서 가장 최근 날짜의 값을 고른다 (순수 함수).

    입력: history, target_date(YYYY-MM-DD)
    출력: (환율, 그 날짜) 또는 골라 쓸 값이 없으면 None
    """
    candidates = [d for d in history if d <= target_date]
    if not candidates:
        return None
    picked_date = max(candidates)
    return history[picked_date], picked_date


def fetch_usd_krw_history(period: str = "15d") -> dict[str, float]:
    """yfinance `KRW=X`의 최근 일별 종가를 {날짜: 환율} dict로 받는다.

    입력: period(yfinance period 문자열)
    출력: {"YYYY-MM-DD": 환율, ...}. 응답이 비어 있으면 빈 dict.
    예외: 네트워크·응답 형식 오류는 그대로 올린다 (호출부가 경고로 모은다)
    """
    import yfinance as yf  # 테스트 환경에서 네트워크 모듈 import를 늦춘다

    raw = yf.Ticker("KRW=X").history(period=period, auto_adjust=False)
    if raw.empty:
        return {}
    out: dict[str, float] = {}
    for ts, row in raw.iterrows():
        close = row.get("Close")
        if close is None or close != close:  # NaN
            continue
        out[ts.date().isoformat()] = float(close)
    return out


def get_usd_krw_rate(as_of_date, fetch_provider=fetch_usd_krw_history) -> FxRateResult:
    """기준일의 원/달러 종가 환율을 구한다 (매번 새로 받고, 실패하면 캐시로 대체).

    입력: as_of_date(date 또는 "YYYY-MM-DD" 문자열), fetch_provider(테스트 주입용 —
         기본은 fetch_usd_krw_history. 인자 없이 불러 {날짜: 환율}을 돌려줘야 한다)
    출력: FxRateResult
    """
    target = as_of_date.isoformat() if hasattr(as_of_date, "isoformat") else str(as_of_date)
    stored = _load_cache()

    try:
        fresh = fetch_provider()
    except Exception as exc:
        fresh = {}
        fetch_error = str(exc)
    else:
        fetch_error = None

    merged = {**stored, **fresh}
    if fresh:
        _save_cache(merged)

    picked = pick_rate_for_date(merged, target)
    if picked is None:
        warning = "환율을 구할 수 없습니다 (캐시도 없음)" + (f" — {fetch_error}" if fetch_error else "")
        return FxRateResult(rate=None, rate_date=None, is_fallback=True, warning=warning)

    rate, rate_date = picked
    if rate_date == target:
        return FxRateResult(rate=rate, rate_date=rate_date, is_fallback=False)

    warning = f"{target} 환율을 새로 받지 못해 직전 캐시 값({rate_date})을 씁니다"
    if fetch_error:
        warning += f" — {fetch_error}"
    return FxRateResult(rate=rate, rate_date=rate_date, is_fallback=True, warning=warning)


def fetch_usd_krw_range(start, end) -> dict[str, float]:
    """yfinance `KRW=X`의 [start, end] 구간 일별 종가를 {날짜: 환율} dict로 받는다 (백테스트용).

    입력: start, end(date 또는 date-like)
    출력: {"YYYY-MM-DD": 환율, ...}
    예외: 네트워크·응답 형식 오류는 그대로 올린다
    """
    import yfinance as yf
    from datetime import timedelta

    raw = yf.Ticker("KRW=X").history(start=start, end=end + timedelta(days=1), auto_adjust=False)
    if raw.empty:
        return {}
    out: dict[str, float] = {}
    for ts, row in raw.iterrows():
        close = row.get("Close")
        if close is None or close != close:
            continue
        out[ts.date().isoformat()] = float(close)
    return out


def fetch_fred_series_range(series_id: str, start, end) -> dict[str, float]:
    """FRED(Federal Reserve Economic Data)의 [start, end] 일별 시계열을 CSV로 받는다
    (API 키 불필요 — P5-5 3번, yfinance KRW=X가 부족한 기간의 환율 보완용).

    입력: series_id(예: "DEXKOUS" — 원/달러), start, end(date 또는 date-like)
    출력: {"YYYY-MM-DD": 값, ...} (결측일·공백 값은 제외)
    예외: 네트워크·응답 형식 오류는 그대로 올린다
    """
    import requests

    resp = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": series_id, "cosd": str(start), "coed": str(end)},
        headers={"User-Agent": "nasdaq100-screener/1.0 (backtest research)"},
        timeout=30,
    )
    resp.raise_for_status()
    out: dict[str, float] = {}
    lines = resp.text.splitlines()
    for line in lines[1:]:  # 첫 줄은 헤더(observation_date,<series_id>)
        parts = line.split(",")
        if len(parts) != 2:
            continue
        d, v = parts[0].strip(), parts[1].strip()
        if not d or not v or v == ".":
            continue
        try:
            out[d] = float(v)
        except ValueError:
            continue
    return out


def merge_fx_with_fallback(primary: dict[str, float], fallback: dict[str, float]) -> tuple[dict[str, float], dict]:
    """primary(yfinance KRW=X)를 우선하고, primary에 없는 날짜만 fallback(FRED)으로 채운다
    (순수 함수, P5-5 3번). 겹치는 기간의 두 자료 차이도 함께 보고한다.

    입력: primary({날짜: 환율}), fallback({날짜: 환율})
    출력: (병합된 {날짜: 환율}, {overlap_days, mean_abs_diff, max_abs_diff, max_abs_diff_date,
          filled_from_fallback_days, filled_dates})
    """
    merged = {**fallback, **primary}  # primary가 있으면 그 값으로 덮어쓴다
    overlap = sorted(set(primary) & set(fallback))
    diffs = [(d, primary[d] - fallback[d]) for d in overlap]
    filled_dates = sorted(set(fallback) - set(primary))
    stats = {
        "overlap_days": len(overlap),
        "mean_abs_diff": round(sum(abs(v) for _, v in diffs) / len(diffs), 2) if diffs else None,
        "max_abs_diff": round(max((abs(v) for _, v in diffs), default=0.0), 2),
        "max_abs_diff_date": max(diffs, key=lambda kv: abs(kv[1]))[0] if diffs else None,
        "filled_from_fallback_days": len(filled_dates),
        "filled_dates": filled_dates,
    }
    return merged, stats


def get_usd_krw_rate_map(dates: list, fetch_provider=fetch_usd_krw_history) -> dict[str, float | None]:
    """여러 날짜의 원/달러 환율을 한 번의 조회로 구한다 (며칠치를 한 번에 처리하는
    engine.daily의 catch-up 실행·engine.backtest용 — 날짜마다 네트워크를 부르지 않는다).

    입력: dates(date 또는 date-like 목록), fetch_provider(테스트 주입용 — 백테스트처럼
         긴 기간이 필요하면 더 넓은 period나 명시적 범위를 받는 provider를 넘긴다)
    출력: {날짜.isoformat(): 환율 또는 (그 날짜 이하 값이 전혀 없으면) None}
    """
    stored = _load_cache()
    try:
        fresh = fetch_provider()
    except Exception:
        fresh = {}

    merged = {**stored, **fresh}
    if fresh:
        _save_cache(merged)

    out: dict[str, float | None] = {}
    for d in dates:
        # pd.Timestamp/datetime은 .isoformat()에 시각까지 붙으므로 날짜만(.date())로 맞춘다.
        target = d.date().isoformat() if hasattr(d, "date") and callable(d.date) else (d.isoformat() if hasattr(d, "isoformat") else str(d))
        picked = pick_rate_for_date(merged, target)
        out[target] = picked[0] if picked else None
    return out
