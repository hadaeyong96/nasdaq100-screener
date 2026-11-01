"""검색량 실적 예고 — 수집(네트워크) (docs/prereg/search_forecast.md 2장).

- 구글 트렌드: trendspy==0.1.6(버전 고정, 소스 확인 2026-11-01 — trends.google.com에만 요청, 파일·하위 프로세스 없음).
  한 요청에 검색어 1개. 요청 사이 대기, 실패하면 30·60·120초 쉬고 다시(3회). 그래도 실패하면 예외 — 호출부가 기록만 한다.
- 위키백과: Wikimedia REST pageviews(월간, user 접속). User-Agent는 저장소 이름과 연락 문구만.
- EDGAR: 분기 매출 사실·공시 목록(data/edgar, SEC_USER_AGENT). 주가: yfinance auto_adjust=False Close.
"""

from __future__ import annotations

import time
from urllib.parse import quote

import pandas as pd

WIKI_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/all-access/user/{article}/monthly/{start}/{end}"


class TrendsClient:
    """trendspy 감싸기: 요청 간격·재시도를 여기서 지킨다(라이브러리 자체 재시도는 1회로 둔다)."""

    def __init__(self, geo: str = "US", timeframe: str = "today 5-y", delay: float = 20.0, retries: int = 3,
                 backoff: float = 30.0, sleep=time.sleep):
        from trendspy import Trends

        self._tr = Trends(request_delay=delay, max_retries=1)
        self.geo, self.timeframe, self.delay, self.retries, self.backoff = geo, timeframe, delay, retries, backoff
        self._sleep = sleep
        self._last = 0.0

    def weekly(self, term: str) -> pd.Series:
        """검색어 하나의 주간 값(인덱스 = 주 시작일, 0~100). 값이 비면 ValueError."""
        err = None
        for k in range(self.retries + 1):
            wait = self.delay - (time.monotonic() - self._last)
            if wait > 0:
                self._sleep(wait)
            try:
                self._last = time.monotonic()
                df = self._tr.interest_over_time([term], timeframe=self.timeframe, geo=self.geo)
                col = term if term in df.columns else df.columns[0]
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                s.index = pd.DatetimeIndex(s.index).tz_localize(None)
                if s.empty:
                    raise ValueError("빈 결과")
                return s
            except Exception as e:  # noqa: BLE001 — 재시도 뒤 호출부로
                err = e
                if k < self.retries:
                    self._sleep(self.backoff * 2 ** k)
        raise RuntimeError(f"트렌드 실패({type(err).__name__}: {str(err)[:120]})")


def wiki_monthly(article: str, user_agent: str, years: int = 5, project: str = "en.wikipedia", today=None) -> pd.Series:
    """위키백과 문서 월간 조회수(최근 years년). 실패하면 예외."""
    import requests

    t = pd.Timestamp(today) if today is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    start = (t - pd.DateOffset(years=years)).strftime("%Y%m01")
    end = (t.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y%m%d")  # 지난달 말까지(완결된 달)
    url = WIKI_URL.format(project=project, article=quote(article, safe=""), start=start, end=end)
    r = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"위키 HTTP {r.status_code}")
    items = r.json().get("items", [])
    return pd.Series({pd.Timestamp(i["timestamp"][:8]): int(i["views"]) for i in items}, dtype="int64").sort_index()


def edgar_bundle(ticker: str, revenue_tags: list[str]) -> dict:
    """EDGAR: 분기 매출 사실(태그 우선순위대로 합침)·최근 공시 목록. 출력: {revenue_facts: [...], filings: [...]}."""
    from data import edgar

    cik = edgar.fetch_ticker_to_cik()[ticker.upper()]
    facts = edgar.fetch_company_facts(cik, force_refresh=True)
    gaap = facts.get("facts", {}).get("us-gaap", {})
    seen, out = set(), []
    for tag in revenue_tags:
        for f in gaap.get(tag, {}).get("units", {}).get("USD", []):
            if not f.get("start") or f.get("form") not in ("10-Q", "10-K", "10-Q/A", "10-K/A"):
                continue
            key = (f["start"], f["end"])
            if key in seen:
                continue
            seen.add(key)
            out.append({"start": f["start"], "end": f["end"], "val": f["val"], "filed": f["filed"]})
    return {"revenue_facts": out, "filings": edgar.fetch_recent_filings(cik)}


def closes(ticker: str, start: str) -> pd.Series:
    """yfinance auto_adjust=False 일봉 Close(분할 반영)."""
    import yfinance as yf

    df = yf.download(ticker, start=start, auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: 가격 없음")
    c = df["Close"]
    c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
    c.index = pd.DatetimeIndex(c.index).tz_localize(None)
    return c.dropna()
