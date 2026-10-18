"""가상 장부 8번 「가치평가 따라하기」 — 라이브 명단·월말 점수 (네트워크). 계산은 core.value_score 그대로.

- 명단(그 월말 기준): **S&P 500 ∪ 나스닥 100** [사용자 결정 2026-10-18 23:12 — 앞으로 기록에는 구역 제한이 필요 없다.
  검증 ③ 판정 장부는 S&P 500 − (나스닥 100 ∪ S&P 100)였다 — docs/results/value_score_zone3.md].
  S&P 500 = 위키 "List of S&P 500 companies" 현재 표 + "Historical components" 변경 표로 그 날짜까지 되돌림.
  나스닥 100 = data.universe_history 역산. 티커는 BRK.B → BRK-B. 금융(SIC 6000~6799) 등 나머지 필터는 그대로(compute_scores).
- 재무: EDGAR companyfacts(SEC User-Agent, 초당 10회 이하) 10-K·10-K/A 사실만, 판단일 전 접수본만(core.value_score.annual_table).
- SIC: SEC submissions (한 번 받은 값은 sic_cache로 재사용).
- 그날 실제 종가: yfinance 종가 × (그날 뒤 ~ 오늘 분할·분사 비율 곱). EPS·주식 수 분할 조정은 흔한 분할 비율만(사전 등록 4-1-5).
네트워크 실패는 조용히 넘기지 않는다 — 종목별 실패를 warnings에 모아 돌려준다(그 종목은 점수 없음 = 판정 불가).
"""

from __future__ import annotations

import io
from datetime import date, timedelta

import numpy as np
import pandas as pd

from core import value_score as vs

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"  # 변경 표(2026-10 기준 이 문서로 옮겨짐)
SP100_URL = "https://en.wikipedia.org/wiki/S%26P_100"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nasdaq100-screener/1.0)"}
NICE = [2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 50, 1.5, 2.5, 1.25, 4 / 3]
MIN_PRICE = 5.0
FORMS = ("10-K", "10-K/A")


class ValueLiveError(RuntimeError):
    """명단을 받지 못했을 때 (점수 전체를 계산할 수 없음)."""


def norm(t: str) -> str:
    return str(t).strip().upper().replace(".", "-")


def is_true_split(r: float) -> bool:
    cands = NICE + [1 / q for q in NICE]
    return any(abs(r / q - 1) < 0.01 for q in cands)


def _tables(url: str) -> list[pd.DataFrame]:
    import requests

    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_html(io.BytesIO(r.content))


def sp500_on(asof: date, tables=None, change_tables=None) -> set[str]:
    """그날 S&P 500: 현재 표에서 asof 뒤 변경을 거꾸로 되돌린다(추가된 것은 빼고 빠진 것은 넣는다)."""
    ts = tables if tables is not None else _tables(SP500_URL)
    cur = next(t for t in ts if "Symbol" in [str(c) for c in t.columns])
    ts = list(ts) + list(change_tables if change_tables is not None else (_tables(SP500_CHANGES_URL) if tables is None else []))
    members = {norm(x) for x in cur["Symbol"]}
    ch = next((t for t in ts if isinstance(t.columns, pd.MultiIndex) and any("Added" in str(c) for c in t.columns.get_level_values(0))), None)
    if ch is None:
        raise ValueLiveError("S&P 500 변경 표를 찾지 못했습니다")
    ch = ch.copy()  # 받은 표를 바꾸지 않는다(같은 표로 여러 날짜를 되돌릴 수 있게)
    ch.columns = [" ".join(str(x) for x in c).strip() for c in ch.columns]
    dcol = next(c for c in ch.columns if c.lower().startswith("date") or "effective" in c.lower())
    add = next(c for c in ch.columns if c.startswith("Added") and "Ticker" in c)
    rem = next(c for c in ch.columns if c.startswith("Removed") and "Ticker" in c)
    ch["_d"] = pd.to_datetime(ch[dcol], errors="coerce")
    for _, r in ch[ch["_d"] > pd.Timestamp(asof)].sort_values("_d", ascending=False).iterrows():
        if isinstance(r[add], str) and r[add].strip():
            members.discard(norm(r[add]))
        if isinstance(r[rem], str) and r[rem].strip():
            members.add(norm(r[rem]))
    return members


def sp100_now(tables=None) -> set[str]:
    ts = tables if tables is not None else _tables(SP100_URL)
    t = next(x for x in ts if "Symbol" in [str(c) for c in x.columns])
    return {norm(x) for x in t["Symbol"]}


def universe_on(asof: date, sp500_tables=None, sp100_tables=None, ndx_fn=None, change_tables=None) -> set[str]:
    """그날 대상 명단: S&P 500 ∪ 나스닥 100 (사용자 결정 2026-10-18 23:12). sp100_tables는 예전 호출 호환용(쓰지 않음)."""
    try:
        sp = sp500_on(asof, sp500_tables, change_tables)
        if ndx_fn is None:
            from data import universe_history as uh

            ndx = {norm(x) for x in uh.point_in_time_universe(asof)}
        else:
            ndx = ndx_fn(asof)
    except ValueLiveError:
        raise
    except Exception as exc:
        raise ValueLiveError(f"가치평가 명단을 받지 못했습니다: {type(exc).__name__}: {exc}") from exc
    if len(sp) < 450:
        raise ValueLiveError(f"S&P 500 명단이 {len(sp)}종목뿐입니다 — 표 형식이 바뀌었을 수 있습니다")
    if len(ndx) < 90:
        raise ValueLiveError(f"나스닥 100 명단이 {len(ndx)}종목뿐입니다")
    return sp | ndx


def _facts_frame(cf: dict) -> pd.DataFrame:
    from data import edgar

    rows = []
    for tax in ("us-gaap",):
        for tag in vs.ALL_TAGS:
            for e in edgar.extract_fact_entries(cf, tax, tag):
                if e.get("form") in FORMS and e.get("filed") and e.get("end"):
                    rows.append({"tag": tag, "start": e.get("start"), "end": e["end"], "val": float(e["val"]),
                                 "filed": e["filed"], "form": e["form"], "unit": e["unit"]})
    return vs.prepare_facts(pd.DataFrame(rows, columns=["tag", "start", "end", "val", "filed", "form", "unit"]))


def _price_and_splits(t: str, asof: date) -> tuple[float | None, pd.Series]:
    """(그날 실제 종가, 흔한 분할 비율 Series — asof 이하). yfinance 한 종목 (batch_prices가 못 받은 종목용)."""
    import yfinance as yf

    tk = yf.Ticker(t)
    h = tk.history(start=(asof - timedelta(days=10)).isoformat(), end=(asof + timedelta(days=1)).isoformat(), auto_adjust=False)
    if h is None or h.empty:
        return None, pd.Series(dtype=float)
    h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
    if pd.Timestamp(asof) not in h.index:
        return None, pd.Series(dtype=float)
    sp = tk.splits
    return _raw_and_true(float(h.at[pd.Timestamp(asof), "Close"]), sp if sp is not None else pd.Series(dtype=float), asof)


def _raw_and_true(close: float, sp: pd.Series, asof: date) -> tuple[float, pd.Series]:
    """캐시·받은 종가(그 뒤 분할 반영) → 그날 실제 종가, 그리고 asof 이하 흔한 분할만."""
    sp = sp[sp > 0] if len(sp) else sp
    if len(sp):
        idx = pd.DatetimeIndex(sp.index)
        sp.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    after = sp[sp.index > pd.Timestamp(asof)] if len(sp) else sp
    raw = close * float(np.prod(after.to_numpy(dtype=float))) if len(after) else close
    true = sp[(sp.index <= pd.Timestamp(asof)) & np.array([is_true_split(float(x)) for x in sp], dtype=bool)] if len(sp) else sp
    return raw, true


def batch_prices(tickers: list[str], asof: date, years: int = 7) -> dict[str, tuple[float, pd.Series]]:
    """yfinance 한 번에: {티커: (그날 실제 종가, 흔한 분할)}. 받지 못한 종목은 빠진다(호출부가 한 종목씩 다시)."""
    import yfinance as yf

    start = (pd.Timestamp(asof) - pd.DateOffset(years=years)).date()
    raw = yf.download(sorted(tickers), start=start.isoformat(), auto_adjust=False, actions=True, group_by="ticker",
                      threads=True, progress=False)
    out = {}
    if raw is None or raw.empty:
        return out
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize() if raw.index.tz is not None else pd.DatetimeIndex(raw.index).normalize()
    for t in tickers:
        try:
            g = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            if pd.Timestamp(asof) not in g.index or pd.isna(g.at[pd.Timestamp(asof), "Close"]):
                continue
            sp = g["Stock Splits"] if "Stock Splits" in g.columns else pd.Series(dtype=float)
            out[t] = _raw_and_true(float(g.at[pd.Timestamp(asof), "Close"]), sp.fillna(0.0).astype(float), asof)
        except KeyError:
            continue
    return out


class _Limiter:
    """여러 스레드가 함께 쓰는 SEC 속도 제한(초당 10회 이하)."""

    def __init__(self, gap: float = 0.11):
        import threading

        self.gap, self.lock, self.last = gap, threading.Lock(), 0.0

    def wait(self) -> None:
        import time

        with self.lock:
            now = time.monotonic()
            if now - self.last < self.gap:
                time.sleep(self.gap - (now - self.last))
            self.last = time.monotonic()


def _sec_get(url: str, ua: str, lim: _Limiter):
    import requests

    lim.wait()
    r = requests.get(url, headers={"User-Agent": ua}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return r.json()


def compute_scores(asof, extra: list[str] | None = None, *, sic_cache: dict | None = None, universe_fn=None,
                   facts_fn=None, price_fn=None, sic_fn=None, cik_map=None) -> tuple[pd.DataFrame, list[str]]:
    """월말 asof 점수표. 출력: (DataFrame(index=티커; total(70점판), peg_value, universe, sic2, n_na, 항목 10개), warnings).

    점수는 명단 ∪ extra(보유 종목) 전부에 매긴다. 대상(universe) = 명단 ∩ 금융(SIC 6000~6799) 아님 ∩ 실제 종가 ≥ 5달러 ∩
    이어진 회계연도 ≥ 3 ∩ 최근 3년 흑자 — 사전 등록 4-1-8. 순위 항목(⑤·⑩)의 기준 집단도 대상.
    facts_fn(ticker) → 사실 표 / price_fn(ticker, asof) → (실제 종가, 흔한 분할) / sic_fn(ticker) → SIC 문자열 (테스트 주입).
    """
    from data import edgar

    asof_d = pd.Timestamp(asof).date()
    warnings: list[str] = []
    members = (universe_fn or universe_on)(asof_d)
    tickers = sorted(members | {norm(x) for x in (extra or [])})
    sic_cache = {} if sic_cache is None else sic_cache
    if facts_fn is None or sic_fn is None:
        cmap = cik_map if cik_map is not None else {norm(k): v for k, v in edgar.fetch_ticker_to_cik(force_refresh=True).items()}
    ms, info = {}, {}
    fetched: dict = {}
    if facts_fn is None or sic_fn is None:  # EDGAR: 스레드 6개, 함께 쓰는 속도 제한(초당 10회 이하)
        from concurrent.futures import ThreadPoolExecutor

        ua, lim = edgar.get_user_agent(), _Limiter()

        def one(t):
            cik = cmap.get(t)
            if cik is None:
                raise ValueError("CIK 없음")
            sic = sic_cache.get(t) if t in sic_cache else None
            if t not in sic_cache and sic_fn is None:
                j = _sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", ua, lim)
                sic = str(j.get("sic")) if j.get("sic") else None
            facts = None if facts_fn is not None else _facts_frame(_sec_get(edgar.COMPANY_FACTS_URL_TMPL.format(cik=cik), ua, lim))
            return t, sic, facts

        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {t: ex.submit(one, t) for t in tickers}
            for t, f in futs.items():
                try:
                    fetched[t] = f.result()
                except Exception as exc:  # noqa: BLE001 — 종목별 실패는 모아서 보고
                    fetched[t] = exc
    prices = {} if price_fn is not None else batch_prices(tickers, asof_d)
    for t in tickers:
        try:
            got = fetched.get(t)
            if isinstance(got, Exception):
                raise got
            if t not in sic_cache:
                sic_cache[t] = sic_fn(t) if sic_fn is not None else got[1]
            facts = facts_fn(t) if facts_fn is not None else got[2]
            if price_fn is not None:
                raw, splits = price_fn(t, asof_d)
            else:
                raw, splits = prices.get(t) or _price_and_splits(t, asof_d)
            if raw is None:
                raise ValueError("그날 종가 없음")
        except Exception as exc:  # noqa: BLE001 — 종목별 실패는 모아서 보고(그 종목은 점수 없음)
            warnings.append(f"{t}: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        m = vs.metrics(vs.annual_table(facts, pd.Timestamp(asof_d)), raw, splits, pd.Timestamp(asof_d))
        sic = sic_cache.get(t)
        fin = sic is not None and str(sic).isdigit() and 6000 <= int(sic) <= 6799
        ms[t] = m
        info[t] = {"universe": t in members and not fin and raw >= MIN_PRICE and m.years >= 3 and m.profitable,
                   "sic2": str(sic).zfill(4)[:2] if sic else None, "price_raw": raw}
    sc = vs.score_day(ms, [t for t in ms if info[t]["universe"]])
    if sc.empty:
        return pd.DataFrame(columns=["total", "peg_value", "universe", "sic2", "n_na"] + vs.ITEMS), warnings
    sc["universe"] = [info[t]["universe"] for t in sc.index]
    sc["sic2"] = [info[t]["sic2"] for t in sc.index]
    sc["price_raw"] = [info[t]["price_raw"] for t in sc.index]
    return sc, warnings
