"""S&P 500(나스닥 100 제외) 2012~2021 종목: 회사 번호(CIK)·이름 변경·같은 회사 확인·파산 표시 (조사만).

docs/research/sp500_universe_check.md "최종 공백" 장의 재료. 수익률·신호 계산 없음. Tiingo 한도를 쓰지 않는다
(EDGAR·yfinance만 사용, SEC 권고대로 초당 10회 이하).

단계 (--step):
- cik: 670종목마다 "S&P에 있던 그 회사"의 CIK를 정한다 → identity.csv
    후보 = 위키 2025 판 현재 표 CIK, SEC 현재 티커 목록 CIK, EDGAR 이름 검색(위키 편출 이름)
    채택 기준 = 편입 기간에 10-K/10-Q를 낸 후보 중 이름이 가장 비슷한 것
- renames: fja 명단에서 같은 날 사라지고 생긴 티커 짝 중 위키 변경 표로 설명되지 않는 것 → 이름 변경 후보,
    EDGAR CIK가 같으면 확정 → renames.csv. 새 티커로 yfinance를 다시 받는다 (캐시 data/cache/sp500/prices/)
- check: 가격이 있는 종목 전체에 시가총액(EDGAR 발행 주식 수 × 분할 되돌린 가격)과 편출 직전 가격 연속성 검사,
    8-K item 1.03(파산) 표시 → final.csv
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import sp500_tiingo_check as tc  # noqa: E402
import sp500_universe_check as su  # noqa: E402

CACHE = su.CACHE
SUB_DIR = CACHE / "submissions"
SEARCH_DIR = CACHE / "edgar_search"
IDENTITY = CACHE / "identity.csv"
RENAMES = CACHE / "renames.csv"
SPLITS = CACHE / "splits_full.json"
FINAL = CACHE / "final.csv"
MCAP_MIN_USD = 1e9  # 이보다 작으면 S&P 500 편입 종목으로 그럴듯하지 않음 (사용자 지시)
MCAP_MAX_USD = 5e12  # 이보다 크면 단위·종목 혼입 의심 (조사용 상한)
_last_call = [0.0]
SEARCH_FAILED: list[str] = []


def _sec_get(url: str, params: dict | None = None) -> dict | None:
    """SEC 요청 (초당 10회 이하). 404면 None, 그 밖의 실패는 예외."""
    import requests

    wait = 0.12 - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    r = requests.get(url, params=params, headers={"User-Agent": ua}, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def submissions(cik: int) -> dict | None:
    """data.sec.gov submissions (옛 공시 파일까지 합쳐 filings 목록 하나로). 캐시."""
    path = SUB_DIR / f"CIK{cik:010d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    d = _sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if d is None:
        return None
    recent = d["filings"]["recent"]
    cols = ["filingDate", "form", "items", "reportDate"]
    rows = {c: list(recent.get(c, [])) for c in cols}
    for f in d["filings"].get("files", []):
        more = _sec_get(f"https://data.sec.gov/submissions/{f['name']}")
        if more:
            for c in cols:
                rows[c] += list(more.get(c, []))
    out = {"cik": cik, "name": d.get("name"), "tickers": d.get("tickers", []),
           "formerNames": d.get("formerNames", []), "filings": rows}
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    return out


def name_search(name: str) -> list[tuple[int, str]]:
    """EDGAR 회사 이름 검색(efts 자동완성). 출력: [(cik, 이름)] 상위 10개. 캐시."""
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:80]
    path = SEARCH_DIR / f"{key}.json"
    if path.exists():
        hits = json.loads(path.read_text(encoding="utf-8"))
    else:
        import requests

        d = None
        for q in (name, re.sub(r"[^A-Za-z0-9 ]+", " ", name).strip()):
            try:
                d = _sec_get("https://efts.sec.gov/LATEST/search-index", {"keysTyped": q})
                break
            except requests.HTTPError:
                continue
        if d is None:
            SEARCH_FAILED.append(name)  # 실패는 모아서 보고한다 (캐시하지 않음 — 다음 실행에서 다시 시도)
            return []
        hits = [(int(h["_id"]), h["_source"]["entity"]) for h in d.get("hits", {}).get("hits", [])][:10]
        SEARCH_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(hits), encoding="utf-8")
    return [(int(c), n) for c, n in hits]


_STOP = {"inc", "corp", "corporation", "co", "company", "the", "plc", "ltd", "limited", "group", "holdings",
         "holding", "incorporated", "class", "a", "b", "c", "de", "new", "sa", "nv", "lp", "llc", "trust"}


def norm_name(s: str) -> str:
    """회사 이름 비교용: 소문자, 기호·흔한 꼬리(inc, corp 등) 제거."""
    s = re.sub(r"\(.*?\)", " ", str(s).lower()).replace("&", " and ")
    return " ".join(w for w in re.findall(r"[a-z0-9]+", s) if w not in _STOP)


def name_sim(a: str, b: str) -> float:
    """두 회사 이름 유사도 0~1 (순수 함수)."""
    a, b = norm_name(a), norm_name(b)
    if not a or not b:
        return 0.0
    if a == b or a.startswith(b + " ") or b.startswith(a + " "):
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def filed_in_period(sub: dict, start: pd.Timestamp, end: pd.Timestamp) -> int:
    """기간 안에 낸 10-K·10-Q 수 (순수 함수)."""
    f = sub["filings"]
    return sum(1 for d, form in zip(f["filingDate"], f["form"])
               if form in ("10-K", "10-Q", "10-K405", "10-KT") and start <= pd.Timestamp(d) <= end)


def wiki_current() -> pd.DataFrame:
    """위키 2025-05-27 판 현재 표: index=티커(yfinance 형식), 열 name, cik."""
    t = pd.read_html(su.WIKI_PATH)[0]
    return pd.DataFrame({"name": t["Security"].values, "cik": t["CIK"].astype(int).values},
                        index=[su.norm(x) for x in t["Symbol"].astype(str)])


def step_cik() -> None:
    """670종목의 회사 CIK를 정한다 → identity.csv."""
    from dotenv import load_dotenv

    from data import edgar

    load_dotenv(ROOT / ".env")  # SEC_USER_AGENT (값은 출력하지 않는다)
    pool = pd.read_csv(tc.YF_RESULT, index_col=0, parse_dates=["first_month", "last_month"])
    wiki = wiki_current()
    reasons = tc.removal_reasons()
    secmap = edgar.fetch_ticker_to_cik()
    rows = []
    for i, (t, r) in enumerate(pool.iterrows()):
        start = r["first_month"] - pd.offsets.MonthBegin(1) - pd.Timedelta(days=120)
        end = r["last_month"] + pd.Timedelta(days=120)  # 편입 1~2달 종목도 10-Q·10-K 2건 이상 잡히게
        expect = wiki.loc[t, "name"] if t in wiki.index else (reasons.loc[t, "removed_name"] if t in reasons.index else "")
        cands: list[tuple[str, int]] = []
        if t in wiki.index:
            cands.append(("위키 현재 표", int(wiki.loc[t, "cik"])))
        c = secmap.get(t.replace("-", "."), secmap.get(t))
        if c is not None:
            cands.append(("SEC 티커 목록", int(c)))
        if expect:
            cands += [("이름 검색", c2) for c2, _ in name_search(str(expect))[:5]]
        seen, scored = set(), []
        for src, cik in cands:
            if cik in seen:
                continue
            seen.add(cik)
            sub = submissions(cik)
            if sub is None:
                continue
            n = filed_in_period(sub, start, end)
            sim = max([name_sim(expect, sub["name"])] + [name_sim(expect, f.get("name", "")) for f in sub["formerNames"]]) if expect else np.nan
            scored.append({"src": src, "cik": cik, "edgar_name": sub["name"], "filings": n, "sim": sim})
        ok = [s for s in scored if s["filings"] >= 2]
        pick = None
        if ok:
            # 출처 우선(위키 표 → SEC 목록 → 이름 검색). SEC 목록·이름 검색은 기대 이름을 알면 유사도 0.6 이상만
            # (SEC 목록은 지금 티커 주인이라, 티커가 다른 회사로 넘어간 경우 그 회사가 걸린다: BBT → Beacon Financial)
            pref = {"위키 현재 표": 0, "SEC 티커 목록": 1, "이름 검색": 2}
            ok = [s for s in ok if s["src"] == "위키 현재 표" or (s["src"] == "SEC 티커 목록" and not expect)
                  or (s["sim"] if s["sim"] == s["sim"] else 0) >= 0.6]
            ok.sort(key=lambda s: (pref[s["src"]], -(s["sim"] if s["sim"] == s["sim"] else 0)))
            pick = ok[0] if ok else None
        sec_cand = next((s for s in scored if s["src"] == "SEC 티커 목록"), None)
        rows.append({
            "ticker": t, "expected_name": expect, "cik": pick["cik"] if pick else None,
            "edgar_name": pick["edgar_name"] if pick else "", "cik_source": pick["src"] if pick else "못 찾음",
            "name_sim": round(pick["sim"], 2) if pick and pick["sim"] == pick["sim"] else None,
            # 지금 SEC 목록에서 이 티커의 주인이 편입 기간에 공시가 없으면 = 티커가 다른 회사로 넘어감
            "sec_ticker_reused": bool(sec_cand and sec_cand["filings"] < 2),
            "sec_ticker_owner": sec_cand["edgar_name"] if sec_cand else "",
        })
        if (i + 1) % 50 == 0:
            print(f"[cik] {i + 1}/{len(pool)}", flush=True)
    out = pd.DataFrame(rows).set_index("ticker")
    out.to_csv(IDENTITY, encoding="utf-8-sig")
    print(f"[cik] CIK 정함 {int(out['cik'].notna().sum())}/{len(out)} · 출처 {out['cik_source'].value_counts().to_dict()}")
    print(f"[cik] SEC 티커가 지금 다른 회사 것: {int(out['sec_ticker_reused'].sum())}")
    print(f"[cik] EDGAR 이름 검색 실패 {len(SEARCH_FAILED)}: {SEARCH_FAILED}")


def step_renames() -> None:
    """fja 명단의 같은 날 사라짐·생김 짝 → 이름 변경 후보, CIK로 확정. 새 티커로 yfinance 재수집."""
    from dotenv import load_dotenv

    from data import edgar

    load_dotenv(ROOT / ".env")
    fja = su.load_fja()
    _, wch = su.load_wiki()
    secmap = edgar.fetch_ticker_to_cik()
    wiki = wiki_current()
    ident = pd.read_csv(IDENTITY, index_col=0)
    yf = pd.read_csv(tc.YF_RESULT, index_col=0)
    targets = set(yf.index[~yf["yf"]])
    sets = [{su.norm(su.base(x)) for x in s} for s in fja["tickers"]]
    dates = list(fja.index)
    pairs = []
    for k in range(1, len(sets)):
        gone, new = sets[k - 1] - sets[k], sets[k] - sets[k - 1]
        if not gone or not new:
            continue
        near = wch[(wch["date"] - dates[k]).abs() <= pd.Timedelta(days=7)]
        gone -= set(near["removed"].dropna())
        new -= set(near["added"].dropna())
        for g in gone:
            for n in new:
                pairs.append((g, n, dates[k].date()))
    rows = []
    for g, n, d in pairs:
        gc = ident.loc[g, "cik"] if g in ident.index else None
        nc = wiki.loc[n, "cik"] if n in wiki.index else secmap.get(n.replace("-", "."), secmap.get(n))
        same = gc is not None and not pd.isna(gc) and nc is not None and int(gc) == int(nc)
        if not same and nc is not None and g not in ident.index:
            same = False
        rows.append({"old": g, "new": n, "date": d, "old_cik": gc, "new_cik": nc, "same_cik": same})
    df = pd.DataFrame(rows)
    # 옛 티커의 CIK를 못 정한 경우만: 새 티커 CIK가 옛 티커 편입 기간에 공시가 있고, 옛 회사 이름을 알면
    # 새 CIK의 현재·옛 이름과도 맞아야 같은 회사로 본다 (대형사는 어느 기간에나 공시가 있어 공시만으로는 지수 교체를 못 거름)
    pool = pd.read_csv(tc.YF_RESULT, index_col=0, parse_dates=["first_month", "last_month"])
    for i, r in df[~df["same_cik"]].iterrows():
        old_known = r["old"] in ident.index and not pd.isna(ident.loc[r["old"], "cik"])
        if old_known or r["old"] not in pool.index or r["new_cik"] is None or pd.isna(r["new_cik"]):
            continue
        sub = submissions(int(r["new_cik"]))
        p = pool.loc[r["old"]]
        if not sub or filed_in_period(sub, p["first_month"] - pd.offsets.MonthBegin(1), p["last_month"]) < 2 \
                or len([x for x in pairs if x[0] == r["old"]]) != 1:
            continue
        expect = ident.loc[r["old"], "expected_name"] if r["old"] in ident.index else None
        if isinstance(expect, str) and expect:
            names = [sub["name"]] + [f.get("name", "") for f in sub["formerNames"]]
            if max(name_sim(expect, n) for n in names) < 0.6:
                continue
        df.loc[i, "same_cik"] = True
    conf = df[df["same_cik"]].drop_duplicates("old")
    # 연쇄(HCP→PEAK→DOC) 따라가기
    chain = dict(zip(conf["old"], conf["new"]))

    def final(t):
        seen = set()
        while t in chain and t not in seen:
            seen.add(t)
            t = chain[t]
        return t

    conf = conf.assign(final_ticker=conf["old"].map(final), target=conf["old"].isin(targets))
    conf.to_csv(RENAMES, index=False, encoding="utf-8-sig")
    print(f"[renames] 후보 짝 {len(df)} · 확정 {len(conf)} · 그중 yfinance 실패 종목 {int(conf['target'].sum())}")
    print(conf[conf["target"]][["old", "new", "final_ticker", "date"]].to_string(index=False))
    # 새 티커로 yfinance (E1 load_prices, 2011-06~2021-12, 캐시 키 = 새 티커)
    import e1_experiments as e1

    e1.PRICE_CACHE = CACHE / "prices"
    news = sorted(set(conf[conf["target"]]["final_ticker"]))
    prices, missing = e1.load_prices(news)
    print(f"[renames] 새 티커 yfinance 있음 {len(prices)}/{len(news)} · 없음 {missing}")


def full_splits(tickers: list[str]) -> dict:
    """yfinance 전체 분할 이력(2022년 이후 포함, 가격을 분할 전 원래 값으로 되돌리는 데 씀). 캐시."""
    import yfinance as yf

    have = json.loads(SPLITS.read_text(encoding="utf-8")) if SPLITS.exists() else {}
    for t in tickers:
        if t in have:
            continue
        try:
            s = yf.Ticker(t).splits
            have[t] = {str(k.date()): float(v) for k, v in s.items()} if s is not None else {}
        except Exception as exc:  # noqa: BLE001 — 실패 종목은 모아서 보고
            have[t] = {"_error": type(exc).__name__}
        SPLITS.write_text(json.dumps(have), encoding="utf-8")
    return have


def raw_price_series(t: str, price_ticker: str, source: str, splits: dict) -> pd.Series | None:
    """분할을 되돌린(그날 실제 거래된) 종가. yfinance Close는 오늘까지의 분할이 반영돼 있어 이후 분할을 곱한다."""
    if source == "tiingo":
        f = tc.TIINGO_DIR / f"{t}.csv"
        if not f.exists():
            return None
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        return df["raw_close"] if len(df) else None
    f = CACHE / "prices" / f"{price_ticker}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    if df.empty:
        return None
    sp = {pd.Timestamp(k): v for k, v in splits.get(price_ticker, {}).items() if not k.startswith("_") and v > 0}
    factor = pd.Series(1.0, index=df.index)
    for d, v in sp.items():
        factor[df.index < d] *= v
    return df["close"] * factor


def shares_series(cik: int) -> pd.Series:
    """EDGAR 주식 수 (기준일 → 주식 수). 여러 종류 주식은 같은 날 합산.

    dei:EntityCommonStockSharesOutstanding(표지)을 먼저 쓰고, 4건 미만이면(HRL처럼 표지 태그가 없는 회사)
    us-gaap CommonStockSharesOutstanding(재무상태표) → WeightedAverageNumberOfSharesOutstandingBasic 순으로 쓴다.
    """
    from data import edgar

    f = edgar.fetch_company_facts(int(cik))
    ents = []
    for tax, tag in (("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding"),
                     ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic")):
        ents = [e for e in edgar.extract_fact_entries(f, tax, tag) if e.get("unit") == "shares"]
        if len({e["end"] for e in ents}) >= 4:
            break
    if not ents:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    df = pd.DataFrame(ents).drop_duplicates(subset=["end", "val", "accn"])
    # 한 공시 안에서는 종류별 주식을 합하고, 같은 기준일을 여러 공시가 보고하면 나중 공시 하나만 쓴다
    per = df.groupby(["end", "accn"]).agg(val=("val", "sum"), filed=("filed", "max")).reset_index()
    s = per.sort_values("filed").groupby("end")["val"].last()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def bankruptcy_dates(cik: int) -> list[str]:
    """8-K item 1.03(파산·법정관리) 공시일 목록."""
    sub = submissions(int(cik))
    if not sub:
        return []
    f = sub["filings"]
    return [d for d, form, it in zip(f["filingDate"], f["form"], f["items"])
            if form.startswith("8-K") and "1.03" in str(it).split(",")]


def continuity(px: pd.Series, removed_on) -> dict:
    """편출 직전 가격 연속성 (순수 함수): 마지막 20거래일 변동계수, 최대 하루 변동, 마지막 날과 편출일 차이."""
    s = px.dropna()
    if s.empty:
        return {}
    tail = s.iloc[-20:]
    out = {"last_day": s.index[-1].date(),
           "tail_cv_pct": round(float(tail.std() / tail.mean() * 100), 2),
           "tail_max_jump_pct": round(float(tail.pct_change().abs().max() * 100), 1)}
    if removed_on is not None and not pd.isna(removed_on):
        out["last_vs_removed_days"] = (s.index[-1] - pd.Timestamp(removed_on)).days
    return out


def step_check() -> None:
    """가격 있는 종목 전체: 시가총액·연속성 검사, 파산 표시, 최종 분류 → final.csv."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    pool = pd.read_csv(tc.YF_RESULT, index_col=0, parse_dates=["first_month", "last_month"])
    ident = pd.read_csv(IDENTITY, index_col=0)
    ren = pd.read_csv(RENAMES) if RENAMES.exists() else pd.DataFrame(columns=["old", "final_ticker"])
    ren_map = dict(zip(ren["old"], ren["final_ticker"]))
    reasons = tc.removal_reasons()
    tlog = json.loads(tc.TIINGO_LOG.read_text(encoding="utf-8")) if tc.TIINGO_LOG.exists() else {}
    # 가격 출처 정하기: yfinance(원래 티커) → yfinance(새 티커) → Tiingo(날짜 대조 "맞음"만 후보)
    src = {}
    for t, r in pool.iterrows():
        if r["yf"]:
            src[t] = ("yfinance", t)
        elif t in ren_map and (CACHE / "prices" / f"{ren_map[t]}.csv").exists() and \
                len(pd.read_csv(CACHE / "prices" / f"{ren_map[t]}.csv")) > 0:
            src[t] = ("yfinance(새 티커)", ren_map[t])
        elif tlog.get(t, {}).get("status") == "있음":
            src[t] = ("tiingo", t)
    yf_tickers = sorted({v[1] for v in src.values() if v[0].startswith("yfinance")})
    print(f"[check] 가격 출처 정함 {len(src)}/{len(pool)} — yfinance 분할 이력 받는 중 {len(yf_tickers)}", flush=True)
    splits = full_splits(yf_tickers)
    rows = []
    for t, r in pool.iterrows():
        rr = reasons.loc[t] if t in reasons.index else None
        # 편입 전 편출 기록은 같은 티커를 쓰던 옛 회사 것이다 (T: 2005년 AT&T Corp 편출) → 쓰지 않는다
        if rr is not None and not pd.isna(rr["date"]) and rr["date"] < r["first_month"] - pd.Timedelta(days=62):
            rr = None
        removed_on = rr["date"] if rr is not None else None
        cik = ident.loc[t, "cik"] if t in ident.index else None
        row = {"ticker": t, "first_month": r["first_month"].date(), "last_month": r["last_month"].date(),
               "in_sp500_now": bool(r["in_sp500_now"]), "cik": cik, "edgar_name": ident.loc[t, "edgar_name"] if t in ident.index else "",
               "wiki_reason": rr["category"] if rr is not None else "", "removed_on": removed_on.date() if removed_on is not None and not pd.isna(removed_on) else None,
               "price_source": src.get(t, ("없음", None))[0]}
        if cik is not None and not pd.isna(cik):
            # 편입 기간 시작 ~ 편출일(없으면 편입 마지막 달) + 1년 안의 1.03만 (AAL: 2011년 AMR 파산은 편입 전)
            b_end = (removed_on if removed_on is not None and not pd.isna(removed_on) else r["last_month"]) + pd.Timedelta(days=365)
            b_start = r["first_month"] - pd.Timedelta(days=62)
            row["bankruptcy_8k_103_all"] = ";".join(bankruptcy_dates(int(cik)))
            row["bankruptcy_8k_103"] = ";".join(d for d in bankruptcy_dates(int(cik)) if b_start <= pd.Timestamp(d) <= b_end)
        px = None
        if t in src:
            px = raw_price_series(t, src[t][1], "tiingo" if src[t][0] == "tiingo" else "yf", splits)
        if px is not None and len(px):
            lo = r["first_month"] - pd.offsets.MonthBegin(1)
            hi = min(r["last_month"], pd.Timestamp(su.END))
            mem = px[(px.index >= lo) & (px.index <= hi)]
            row["price_first"], row["price_last"] = px.index.min().date(), px.index.max().date()
            row["date_check"] = tc.judge_period(r["first_month"], r["last_month"], px.index.min(), px.index.max(), removed_on)
            if cik is not None and not pd.isna(cik) and len(mem):
                sh = shares_series(int(cik))
                sh = sh[(sh.index >= lo - pd.Timedelta(days=200)) & (sh.index <= hi)]
                if len(sh):
                    pm = mem.reindex(mem.index.union(sh.index)).ffill().reindex(sh.index).dropna()
                    mc = (sh.reindex(pm.index) * pm).dropna()
                    mc = mc[mc.index >= lo - pd.Timedelta(days=100)]
                    if len(mc):
                        row["mcap_median_b"] = round(float(mc.median()) / 1e9, 2)
                        row["mcap_min_b"] = round(float(mc.min()) / 1e9, 2)
            row.update(continuity(px[px.index <= hi + pd.Timedelta(days=40)], removed_on))
        rows.append(row)
    out = pd.DataFrame(rows).set_index("ticker")

    def suspect(x) -> str:
        why = []
        if x["price_source"] == "없음":
            return ""
        if x.get("date_check") not in ("맞음", None) and isinstance(x.get("date_check"), str):
            why.append(f"날짜 {x['date_check']}")
        if pd.isna(x.get("mcap_median_b")):
            why.append("시가총액 계산 못 함")
        elif x["mcap_median_b"] * 1e9 < MCAP_MIN_USD:
            why.append(f"시가총액 {x['mcap_median_b']}B < 1B")
        elif x["mcap_median_b"] * 1e9 > MCAP_MAX_USD:
            why.append(f"시가총액 {x['mcap_median_b']}B 과다")
        if x["wiki_reason"] == "인수·합병" and not pd.isna(x.get("last_vs_removed_days")) and abs(x["last_vs_removed_days"]) > 10 \
                and x["removed_on"] is not None and pd.Timestamp(x["removed_on"]) <= pd.Timestamp(su.END):
            why.append(f"마지막 거래일-편출일 {int(x['last_vs_removed_days']):+d}일")
        if x["wiki_reason"] == "인수·합병" and not pd.isna(x.get("tail_max_jump_pct")) and x["tail_max_jump_pct"] > 30 \
                and x["removed_on"] is not None and pd.Timestamp(x["removed_on"]) <= pd.Timestamp(su.END):
            why.append(f"편출 직전 하루 {x['tail_max_jump_pct']}% 변동")
        return "; ".join(why)

    out["suspect"] = out.apply(suspect, axis=1)

    def why_gone(x) -> str:
        if x["in_sp500_now"]:
            return "지금도 S&P"
        if isinstance(x.get("bankruptcy_8k_103"), str) and x["bankruptcy_8k_103"]:
            return "파산"
        if x["wiki_reason"] == "인수·합병":
            return "인수·합병"
        if x["wiki_reason"] == "시가총액 등(계속 거래)":
            return "지수 편출만"
        if x["ticker_renamed"]:
            return "이름 변경"
        return "불명"

    out["ticker_renamed"] = out.index.isin(ren_map.keys())
    out["why_gone"] = out.reset_index().apply(why_gone, axis=1).values
    out.to_csv(FINAL, encoding="utf-8-sig")
    print(f"[check] 저장 {FINAL.name}: {len(out)}행")


def gap_status(price_source: str, date_check) -> str:
    """종목 하나의 가격 상태 (순수 함수): 있음 | 공백(가격 없음) | 공백(다른 회사 제외) | 공백(일부만)."""
    if price_source == "없음":
        return "공백(가격 없음)"
    if date_check == "다른 회사 의심":
        return "공백(다른 회사 제외)"
    if date_check in ("앞이 빔", "끝이 이름"):
        return "공백(일부만)"
    return "있음"


def gap_reason(why_gone: str) -> str:
    """공백 종목의 사라진 이유를 지시문 4분류(+ 보조 2분류)로 (순수 함수)."""
    return {"인수·합병": "인수·합병", "지수 편출만": "지수 편출만", "파산": "파산",
            "지금도 S&P": "지금도 S&P(티커 문제)", "이름 변경": "이름 변경(새 티커도 실패)"}.get(why_gone, "불명")


def step_summary() -> None:
    """final.csv → 최종 공백 표 (네트워크 없음) → final_gap.csv와 요약 출력."""
    d = pd.read_csv(FINAL, index_col=0)
    d["status"] = [gap_status(s, c) for s, c in zip(d["price_source"], d["date_check"])]
    d["gap_reason"] = d["why_gone"].map(gap_reason)
    d.to_csv(CACHE / "final_gap.csv", encoding="utf-8-sig")
    n = len(d)
    print(f"[summary] 전체 {n}")
    print(d["price_source"].value_counts().to_string())
    vc = d["status"].value_counts()
    print(vc.to_string())
    gap = d[d["status"] != "있음"]
    print(f"[summary] 공백 {len(gap)} ({len(gap) / n:.1%})")
    print(pd.crosstab(gap["gap_reason"], gap["status"], margins=True).to_string())
    with pd.option_context("display.width", 250, "display.max_colwidth", 50, "display.max_rows", 500):
        cols = ["price_source", "edgar_name", "wiki_reason", "removed_on", "price_first", "price_last",
                "date_check", "mcap_median_b", "bankruptcy_8k_103", "suspect"]
        print("── 같은 회사 의심 (날짜 판정이 맞음이 아니거나 시가총액 1B 미만) ──")
        sus = d[(d["price_source"] != "없음") & ((d["date_check"] != "맞음") | (d["mcap_median_b"] < 1))]
        print(sus[cols].to_string())
        print("── 공백 전체 ──")
        print(gap[["status", "gap_reason", "wiki_reason", "removed_on", "edgar_name", "bankruptcy_8k_103"]].to_string())
        print("── 파산 표시 (편입 기간 + 1년 안 8-K 1.03) ──")
        b = d[d["bankruptcy_8k_103"].fillna("") != ""]
        print(b[["status", "in_sp500_now", "why_gone", "bankruptcy_8k_103"]].to_string())


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["cik", "renames", "check", "summary"], required=True)
    a = ap.parse_args()
    {"cik": step_cik, "renames": step_renames, "check": step_check, "summary": step_summary}[a.step]()


if __name__ == "__main__":
    main()
