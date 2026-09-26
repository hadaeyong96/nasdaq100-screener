"""시장 온도 지표(CNN 공포·탐욕 지수 + FRED 5개) 수집 + 캐시 (P3.8).

표시 전용이다 — 이 모듈이 반환하는 값은 어떤 매매 신호·필터·수량 계산에도 쓰이지
않는다(docs/p3_8_instructions.md, P5-2에서 국면 필터가 성과를 나쁘게 만들었기 때문).

- FRED 지표(DGS10·T10Y2Y·BAMLH0A0HYM2·DFEDTARU·DEXKOUS·VIXCLS)는
  data.fx.fetch_fred_series_range를 그대로 재사용해 받고 `data/cache/fred/{code}.json`에
  캐시한다.
- CNN 공포·탐욕 지수는 공식 API가 없는 비공식 자료라 **하루 1회만** 요청한다. 실패하면
  캐시 값 + "지연" 표시, 캐시의 최신 값도 3일 넘게 오래됐으면 이 함수가 호출부에
  `use_vix_fallback=True`를 돌려줘 VIX로 대체하게 한다(3일 이내면 "지연" 배지만).
  받은 값은 `data/cache/fear_greed.csv`에 날짜별로 계속 쌓는다(연구용, CNN 이용 약관상
  개인 참고용으로만 쓰고 보고서 외부 공개는 하지 않는다).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
FRED_CACHE_DIR = DATA_DIR / "cache" / "fred"
FEAR_GREED_CSV = DATA_DIR / "cache" / "fear_greed.csv"

CNN_FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{start}"
_CNN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


# ── FRED 지표 (캐시) ─────────────────────────────────────────────────────────


def _fred_cache_path(series_id: str) -> Path:
    return FRED_CACHE_DIR / f"{series_id}.json"


def _load_fred_cache(series_id: str) -> dict[str, float]:
    path = _fred_cache_path(series_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_fred_cache(series_id: str, history: dict[str, float]) -> None:
    FRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _fred_cache_path(series_id).write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")


def fetch_fred_indicator(series_id: str, as_of: date, lookback_days: int = 400, fetch_provider=None) -> dict:
    """FRED 지표 하나를 [as_of - lookback_days, as_of] 받는다 (캐시 우선 병합).

    입력: series_id(예: "DGS10"), as_of(보고서 기준일), lookback_days(1년+여유),
         fetch_provider(테스트 주입용 — 기본은 data.fx.fetch_fred_series_range)
    출력: {"series": {날짜: 값, ...}(정렬됨), "warning": str|None}
         네트워크 실패는 조용히 넘기지 않고 warning에 남기되, 캐시가 있으면 그 값으로 계속한다.
    """
    from data import fx as fxmod

    fetch_provider = fetch_provider or fxmod.fetch_fred_series_range
    cached = _load_fred_cache(series_id)
    start = as_of - timedelta(days=lookback_days)
    warning = None
    try:
        fresh = fetch_provider(series_id, start, as_of)
    except Exception as exc:
        fresh = {}
        warning = f"{series_id} 조회 실패(캐시 값 사용): {exc}"

    merged = {**cached, **fresh}
    if fresh:
        _save_fred_cache(series_id, merged)
    if not merged:
        warning = warning or f"{series_id}: 받은 값도 캐시도 없음"

    series = {d: v for d, v in sorted(merged.items()) if d >= start.isoformat()}
    return {"series": series, "warning": warning}


# ── CNN 공포·탐욕 지수 ───────────────────────────────────────────────────────


def _load_fear_greed_cache() -> dict[str, float]:
    if not FEAR_GREED_CSV.exists():
        return {}
    out: dict[str, float] = {}
    for line in FEAR_GREED_CSV.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) != 2:
            continue
        d, v = parts
        try:
            out[d.strip()] = float(v.strip())
        except ValueError:
            continue
    return out


def _append_fear_greed_cache(new_values: dict[str, float]) -> None:
    """새로 받은 (날짜: 값)을 기존 CSV와 합쳐(중복 날짜는 새 값 우선) 다시 쓴다."""
    if not new_values:
        return
    existing = _load_fear_greed_cache()
    merged = {**existing, **new_values}
    FEAR_GREED_CSV.parent.mkdir(parents=True, exist_ok=True)
    lines = ["date,value"] + [f"{d},{v}" for d, v in sorted(merged.items())]
    FEAR_GREED_CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fetch_fear_greed_cnn(start: date, fetch_provider=None) -> dict[str, float]:
    """CNN 비공식 API에서 공포·탐욕 지수 이력을 받는다.

    입력: start(이 날짜부터), fetch_provider(테스트 주입용 — 인자 없이 불러
         {"date_iso": value, ...}를 돌려주는 호출 가능 객체)
    출력: {"YYYY-MM-DD": 값, ...}
    예외: 네트워크·응답 형식 오류는 그대로 올린다 (호출부가 캐시로 대체)
    """
    if fetch_provider is not None:
        return fetch_provider()

    import requests

    resp = requests.get(CNN_FEAR_GREED_URL.format(start=start.isoformat()), headers=_CNN_HEADERS, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    points = payload.get("fear_and_greed_historical", {}).get("data", [])
    out: dict[str, float] = {}
    for p in points:
        ts_ms, val = p.get("x"), p.get("y")
        if ts_ms is None or val is None:
            continue
        d = date.fromtimestamp(ts_ms / 1000).isoformat()
        out[d] = float(val)
    return out


def get_fear_greed(as_of: date, lookback_days: int = 400, fetch_provider=None, stale_fallback_days: int = 3) -> dict:
    """오늘 카드에 쓸 공포·탐욕 지수를 구한다 (하루 1회 요청, 실패 시 캐시 → 3일 넘으면 VIX 대체 지시).

    출력: {"value": float|None, "as_of": str|None, "series_1y": [값,...], "is_fallback": bool,
          "use_vix_fallback": bool(캐시도 3일 넘게 오래됐으면 True — 호출부가 VIX로 대체),
          "warning": str|None}
    """
    cached = _load_fear_greed_cache()
    already_fetched_today = as_of.isoformat() in cached

    warning = None
    if not already_fetched_today:
        start = as_of - timedelta(days=lookback_days)
        try:
            fresh = fetch_fear_greed_cnn(start, fetch_provider=fetch_provider)
        except Exception as exc:
            fresh = {}
            warning = f"CNN 공포·탐욕 지수 조회 실패(캐시 값 사용): {exc}"
        if fresh:
            _append_fear_greed_cache(fresh)
            cached = {**cached, **fresh}

    if not cached:
        return {"value": None, "as_of": None, "series_1y": [], "is_fallback": True, "use_vix_fallback": True, "warning": warning or "공포·탐욕 지수를 구할 수 없음"}

    latest_date = max(d for d in cached if d <= as_of.isoformat())
    is_fallback = latest_date != as_of.isoformat()
    gap_days = (date.fromisoformat(as_of.isoformat()) - date.fromisoformat(latest_date)).days
    use_vix_fallback = gap_days > stale_fallback_days

    one_year_ago = (as_of - timedelta(days=365)).isoformat()
    series_1y = [v for d, v in sorted(cached.items()) if one_year_ago <= d <= as_of.isoformat()]

    if is_fallback and warning is None:
        warning = f"공포·탐욕 지수 최신 값이 {latest_date}(지연)"

    return {
        "value": cached[latest_date], "as_of": latest_date, "series_1y": series_1y,
        "is_fallback": is_fallback, "use_vix_fallback": use_vix_fallback, "warning": warning,
    }
