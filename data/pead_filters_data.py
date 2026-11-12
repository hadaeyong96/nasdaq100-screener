"""실적급등 보완 필터 탐색 — 입출력 (docs/research/pead_filters_plan.md).

- 반응일 고가·저가·종가(F1): yfinance(auto_adjust=False) 또는 Tiingo(raw, 무료 키 TIINGO_API_KEY) → 캐시 CSV
- XBRL 사실(F4·F5): data/edgar.fetch_company_facts 캐시 → 태그별 표
- 같은 분기 10-Q/10-K Accepted(동부): 공시 색인 페이지 → 캐시 JSON
실패는 조용히 넘기지 않고 (이름, 사유) 목록으로 돌려준다. 키·User-Agent 값은 출력하지 않는다.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from core import pead_filters as pf

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "pead_filters"
HL_DIRS = {"main": CACHE / "hl", "sealed": CACHE / "hl_sealed"}
HL_RANGE = {"main": ("2011-06-01", "2022-01-01"), "sealed": ("2021-01-01", "2026-10-01")}
Q_ACCEPTED = CACHE / "q_accepted.json"


def _hl_frame(h: pd.DataFrame, cols: tuple[str, str, str]) -> pd.DataFrame:
    out = pd.DataFrame({"high": h[cols[0]], "low": h[cols[1]], "close": h[cols[2]]})
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    return out.sort_index()


def fetch_hl_yf(sym: str, period: str = "main") -> pd.DataFrame | None:
    """yfinance auto_adjust=False 고가·저가·종가(캐시). 빈 응답이면 빈 캐시를 남기고 None."""
    d = HL_DIRS[period]
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"yf_{sym}.csv"
    if not p.exists():
        import yfinance as yf

        try:
            h = yf.Ticker(sym).history(start=HL_RANGE[period][0], end=HL_RANGE[period][1], auto_adjust=False)
        except Exception:  # noqa: BLE001 — 실패는 빈 캐시로 남기고 호출 쪽이 보고
            h = None
        if h is None or h.empty:
            pd.DataFrame(columns=["high", "low", "close"]).to_csv(p)
        else:
            _hl_frame(h, ("High", "Low", "Close")).to_csv(p)
    out = pd.read_csv(p, index_col=0, parse_dates=True)
    return out if len(out) else None


def fetch_hl_tiingo(t: str, key: str, period: str = "main") -> tuple[pd.DataFrame | None, str]:
    """Tiingo raw 고가·저가·종가(캐시). 출력: (표 또는 None, 실패 사유). 429면 캐시를 남기지 않는다(다시 실행 시 이어 받음)."""
    import requests

    d = HL_DIRS[period]
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"ti_{t}.csv"
    if not p.exists():
        a, b = HL_RANGE[period]
        try:  # 요청 URL에 토큰이 들어 있어 예외 메시지를 그대로 내보내지 않는다
            r = requests.get(f"https://api.tiingo.com/tiingo/daily/{t}/prices",
                             params={"startDate": a, "endDate": b, "token": key, "format": "json"}, timeout=30)
        except requests.RequestException as exc:
            return None, f"Tiingo 연결 실패({type(exc).__name__})"
        time.sleep(0.5)
        if r.status_code == 429:
            return None, "Tiingo 한도(429)"
        js = r.json() if r.status_code == 200 else None
        if not isinstance(js, list) or not js:
            pd.DataFrame(columns=["high", "low", "close"]).to_csv(p)
            return None, f"Tiingo HTTP {r.status_code}"
        h = pd.DataFrame([{"date": x["date"][:10], "high": x["high"], "low": x["low"], "close": x["close"]} for x in js])
        _hl_frame(h.set_index(pd.to_datetime(h["date"])), ("high", "low", "close")).to_csv(p)
    out = pd.read_csv(p, index_col=0, parse_dates=True)
    return (out, "") if len(out) else (None, "Tiingo 자료 없음")


def tiingo_key() -> str:
    """.env의 TIINGO_API_KEY (값은 출력하지 않는다). 없으면 빈 문자열."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return os.environ.get("TIINGO_API_KEY", "").strip()


def load_tags(cik: int, ua: str | None = None) -> dict[str, pd.DataFrame]:
    """CIK → {태그: fact_frame} (매출 태그 + 영업이익, us-gaap USD). 없는 태그는 빠진다. 네트워크 실패는 예외 그대로."""
    from data import edgar

    j = edgar.fetch_company_facts(int(cik), user_agent=ua)
    g = j.get("facts", {}).get("us-gaap", {})
    out = {}
    for t in pf.REVENUE_TAGS + (pf.OPINC_TAG,):
        u = g.get(t, {}).get("units", {}).get("USD")
        if u:
            out[t] = pf.fact_frame(u)
    return out


def q_accepted_cache() -> dict[str, str]:
    """accn → Accepted('YYYY-MM-DD HH:MM:SS', 동부)."""
    return json.loads(Q_ACCEPTED.read_text(encoding="utf-8")) if Q_ACCEPTED.exists() else {}


def fetch_q_accepted(items: list[tuple[int, str]], ua: str) -> list[tuple[str, str]]:
    """(cik, accn) 목록 중 캐시에 없는 것의 색인 페이지 Accepted를 받는다(초당 8회 이하). 출력: 실패 (accn, 사유)."""
    import requests

    from scripts import sw1_data_check as dc

    cache = q_accepted_cache()
    failed = []
    for cik, acc in items:
        if acc in cache:
            continue
        u = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{acc}-index.htm"
        try:
            r = requests.get(u, headers={"User-Agent": ua}, timeout=30)
        except requests.RequestException as exc:
            failed.append((acc, type(exc).__name__))
            continue
        finally:
            time.sleep(0.125)
        v = dc.parse_index_accepted(r.text) if r.status_code == 200 else None
        if v:
            cache[acc] = v
        else:
            failed.append((acc, f"HTTP {r.status_code}"))
    CACHE.mkdir(parents=True, exist_ok=True)
    Q_ACCEPTED.write_text(json.dumps(cache), encoding="utf-8")
    return failed


def q_accn_of(events: pd.DataFrame, cik: int, report_date: str, q_filing_date: str) -> str | None:
    """events.csv(form·filing_date·report_date·accn·cik) → 그 실적 발표와 같은 분기 10-Q/10-K accn. 없으면 None."""
    e = events[(events["cik"] == int(cik)) & events["form"].isin(["10-Q", "10-K"])
               & (events["report_date"] == report_date) & (events["filing_date"] == q_filing_date)]
    return None if e.empty else str(e.sort_values("accn")["accn"].iloc[0])
