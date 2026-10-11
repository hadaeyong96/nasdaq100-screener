"""SW1(실적 발표 후 흐름, PEAD) 데이터 가능성 조사 — 일회성 점검 스크립트 (조사만, 수익 계산 금지).

단계:
  --step sp100   위키 "S&P 100" 옛 판으로 2011-12~2021-12 매달 명단 → data/cache/sw1/sp100_monthly.csv
  --step events  나스닥 100(시점별) + S&P 100 합집합의 EDGAR 제출 기록 → 8-K 2.02·10-Q·10-K 날짜와 acceptance 시각
  --step prices  거래량 있는 일봉 (data/cache/backtest에 없는 종목만 yfinance로 받음)
  --step tiingo  CIK는 있는데 가격이 없는 종목만 Tiingo → 날짜·시가총액으로 같은 회사 확인 (price_check.csv)
  --step accepted  실적 발표마다 EDGAR 공시 색인 페이지의 Accepted(동부시간)
  --step check   표본 20종목 EPS 계산 가능 비율, acceptance 대조, 연도별 신호 건수 (발표 반응일까지만 본다)

신호 판정은 반응일(발표 뒤 첫 거래 세션)의 종가 변화·거래량까지만 쓰고, 그 뒤 가격은 읽지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "cache" / "sw1"
SUB_CACHE = ROOT / "data" / "cache" / "edgar" / "submissions"
WIKI_CACHE = CACHE / "wiki_sp100"
PRICE_CACHE = CACHE / "prices"
BT_CACHE = ROOT / "data" / "cache" / "backtest"
E1_PRICE_CACHE = ROOT / "data" / "cache" / "e1" / "prices"
SP500_HIST = ROOT / "data" / "cache" / "sp500" / "fja05680_updated.csv"
SP500_IDENTITY = ROOT / "data" / "cache" / "sp500" / "identity.csv"
SP500_RENAMES = ROOT / "data" / "cache" / "sp500" / "renames.csv"
SP500_TIINGO = ROOT / "data" / "cache" / "sp500" / "tiingo"
# 같은 회사(같은 CIK)의 티커 변경 중 S&P 500 점검 renames.csv에 없는 나스닥 100 쪽 것
EXTRA_RENAMES = {"PCLN": "BKNG", "KFT": "MDLZ", "WAG": "WBA"}
SP500_FINAL_GAP = ROOT / "data" / "cache" / "sp500" / "final_gap.csv"
NDX_WIKITEXT = ROOT / "data" / "cache" / "nasdaq100_changes_wikitext.txt"
SEARCH_CACHE = CACHE / "edgar_search"
TIINGO_CACHE = CACHE / "tiingo"
PRICE_CHECK = CACHE / "price_check.csv"
NAME_SIM_MIN = 0.6  # S&P 500 점검과 같은 기준
# 이름 검색을 쓰지 않는 종목 (사람 확인). CA: 검색 상위 10개에 CA, Inc.가 없고, 'Technologies'만 겹친
# 다른 회사(Data Call Technologies 0.81, Cavitation Technologies 0.79)가 유사도 기준을 넘었다
NAME_SEARCH_SKIP = {"CA": "검색 결과에 CA, Inc. 없음 — 다른 회사만 걸림"}
# 시가총액 확인만 못 넘었지만 사람이 같은 회사로 확인한 가격 (S&P 500 점검 WBA와 같은 경우):
# WAG Tiingo 가격 = Walgreen (2012-06 29.93, 2014-06 72.01). 시총 0은 2014년 새 CIK라 그 전 주식 수가 없어서
PRICE_KEEP = {"WAG": "Walgreen 가격 확인, 시총 0은 새 CIK 주식 수 문제"}
MCAP_MIN_USD = 1e9  # S&P 500 점검과 같은 기준 (시가총액 중앙값이 이보다 작으면 다른 회사 의심)

FIRST_MONTH, LAST_MONTH = "2011-12", "2021-12"
START, SEAL = pd.Timestamp("2012-01-01"), pd.Timestamp("2021-12-31")
SAMPLE_SEED, SAMPLE_N = 20261009, 20
WIKI_API = "https://en.wikipedia.org/w/api.php"
SUB_URL = "https://data.sec.gov/submissions/{name}"

# 신호 조건 (지시문 4번 그대로)
RET_MIN, EXCESS_MIN, VOL_MULT, VOL_WINDOW = 0.05, 0.05, 2.0, 20


# ── 순수 함수 ────────────────────────────────────────────────────────────────


def parse_sp100_wikitext(txt: str) -> set[str]:
    """위키 S&P 100 문서 원문 → 명단 표의 티커 집합 (yfinance 형식, BRK.B → BRK-B). 표가 없으면 빈 집합."""
    i = txt.find("! Symbol")
    if i < 0:
        return set()
    j = txt.find("|}", i)
    body = txt[i: j if j > 0 else None]
    found = re.findall(r"\|-\s*\n\|\s*(?:\[\[[^|\]]*\|)?\s*([A-Z]{1,5}(?:[.\-/][A-Z]{1,2})?)\s*(?:\]\])?\s*(?:\n|\|\|)", body)
    return {t.replace(".", "-").replace("/", "-") for t in found}


def acceptance_to_et(raw: str) -> pd.Timestamp:
    """EDGAR submissions의 acceptanceDateTime → 미국 동부 벽시계 시각 (tz 없음).

    AAPL 등 일부 공시는 'Z'가 붙어 있어도 실제 UTC가 아니라 공시 색인 페이지의 Accepted(동부시간)보다
    여름 8시간·겨울 10시간 늦다(동부→UTC 변환이 두 번 들어간 값). 이 함수는 그 경우를 되돌린다.
    다만 다른 공시는 진짜 UTC라서(무작위 12건 중 6건) 이 값만으로는 시각을 정할 수 없다 →
    신호 판정에는 색인 페이지 값(step_accepted)을 쓴다. 이 함수는 대조표용으로만 남긴다.
    입력: '2021-10-29T00:30:23.000Z' → 출력: Timestamp('2021-10-28 16:30:23')
    """
    t = pd.Timestamp(raw)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    once = t.tz_convert("America/New_York").tz_localize(None)
    return once.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)


def session_of(et: pd.Timestamp) -> str:
    """동부시간 → '장전'(09:30 전) | '장중'(16:00 전) | '장후'."""
    hm = et.hour * 60 + et.minute
    if hm < 9 * 60 + 30:
        return "장전"
    if hm < 16 * 60:
        return "장중"
    return "장후"


def reaction_day(et: pd.Timestamp, days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """발표 시각 → 반응일 (2026-10-09 결정 1). 장전이면 그날(휴장이면 다음 거래일), 장중·장후면 다음 거래일."""
    d = et.normalize()
    pos = days.searchsorted(d, side="left" if session_of(et) == "장전" else "right")
    return days[pos] if pos < len(days) else None


def next_trading_day(d: pd.Timestamp, days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """제출일 다음 거래일 (지시문 문구 그대로의 변형: 시각을 보지 않음)."""
    pos = days.searchsorted(pd.Timestamp(d).normalize(), side="right")
    return days[pos] if pos < len(days) else None


def signal_on(px: pd.DataFrame, bench_close: pd.Series, d0: pd.Timestamp) -> dict | None:
    """반응일 d0의 신호 판정. d0 이후 행은 쓰지 않는다 (미래 데이터 금지).

    입력: px(index=거래일, close·volume), bench_close(QQQ 종가), d0
    출력: {"ret","excess","vol_ratio","signal"} 또는 자료 부족이면 None
    """
    h = px.loc[:d0]
    b = bench_close.loc[:d0]
    if len(h) < VOL_WINDOW + 2 or h.index[-1] != d0 or len(b) < 2 or b.index[-1] != d0:
        return None
    c0, c1 = float(h["close"].iloc[-1]), float(h["close"].iloc[-2])
    vol_avg = float(h["volume"].iloc[-VOL_WINDOW - 1:-1].mean())
    if not (c1 > 0) or not (vol_avg > 0):
        return None
    ret = c0 / c1 - 1
    bret = float(b.iloc[-1]) / float(b.iloc[-2]) - 1
    ratio = float(h["volume"].iloc[-1]) / vol_avg
    return {"ret": ret, "excess": ret - bret, "vol_ratio": ratio,
            "signal": bool(ret >= RET_MIN and ret - bret >= EXCESS_MIN and ratio >= VOL_MULT)}


def submissions_rows(sub_jsons: list[dict]) -> pd.DataFrame:
    """submissions JSON(본문 recent + 추가 파일들) → 행 표 (form, filing_date, report_date, acceptance, items)."""
    rows = []
    for j in sub_jsons:
        f = j["filings"]["recent"] if "filings" in j else j
        for k in range(len(f["form"])):
            rows.append({"form": f["form"][k], "filing_date": f["filingDate"][k], "report_date": f["reportDate"][k],
                         "acceptance": f["acceptanceDateTime"][k], "items": f["items"][k] if "items" in f else "",
                         "accn": f["accessionNumber"][k]})
    return pd.DataFrame(rows).drop_duplicates("accn") if rows else pd.DataFrame(
        columns=["form", "filing_date", "report_date", "acceptance", "items", "accn"])


def pick_events(rows: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """제출 기록 → 8-K Item 2.02와 10-Q·10-K(정정 제외)만, 제출일 [start, end]."""
    if rows.empty:
        return rows.assign(kind=pd.Series(dtype=str))
    r = rows[(rows["filing_date"] >= start) & (rows["filing_date"] <= end)].copy()
    is202 = r["form"].isin(["8-K"]) & r["items"].fillna("").str.split(",").apply(lambda xs: "2.02" in [x.strip() for x in xs])
    isq = r["form"].isin(["10-Q", "10-K"])
    r = r[is202 | isq].copy()
    r["kind"] = r["form"].where(isq, "8-K 2.02")
    return r


_STOP = {"inc", "corp", "corporation", "co", "company", "the", "plc", "ltd", "limited", "group", "holdings",
         "holding", "incorporated", "class", "a", "b", "c", "de", "new", "sa", "nv", "lp", "llc", "trust"}


def norm_name(x: str) -> str:
    """회사 이름 비교용: 소문자, 기호·흔한 꼬리(inc, corp 등) 제거 (S&P 500 점검과 같음)."""
    x = re.sub(r"\(.*?\)", " ", str(x).lower()).replace("&", " and ")
    return " ".join(w for w in re.findall(r"[a-z0-9]+", x) if w not in _STOP)


def name_sim(a: str, b: str) -> float:
    """두 회사 이름 유사도 0~1 (S&P 500 점검과 같음)."""
    import difflib

    a, b = norm_name(a), norm_name(b)
    if not a or not b:
        return 0.0
    if a == b or a.startswith(b + " ") or b.startswith(a + " "):
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _link_text(cell: str) -> str:
    """'[[A|B]]' → 'B', '[[A]]' → 'A', 그 밖은 그대로."""
    m = re.search(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", cell)
    return (m.group(2) or m.group(1)).strip() if m else cell.strip()


def parse_ticker_names(txt: str) -> dict[str, set[str]]:
    """위키 표 원문에서 '| 티커' 줄 바로 다음 칸의 회사 이름 (나스닥 100 변경 표·S&P 100 명단 표 공통)."""
    out: dict[str, set[str]] = {}
    lines = txt.splitlines()
    for i, line in enumerate(lines[:-1]):
        m = re.fullmatch(r"\|\s*([A-Z]{1,5}(?:[.\-][A-Z]{1,2})?)\s*", line)
        nxt = lines[i + 1]
        if m and nxt.startswith("|") and not nxt.startswith(("|-", "|}")):
            name = _link_text(nxt[1:])
            if name and not re.fullmatch(r"[A-Z]{1,5}", name):
                out.setdefault(m.group(1).replace(".", "-"), set()).add(name)
    return out


def judge_period(first: pd.Timestamp, last: pd.Timestamp, p_first, p_last, tol_days: int = 10) -> str:
    """가격 기간이 명단 기간과 맞는지 (S&P 500 점검 judge_period와 같은 기준, 편출일 대신 명단 마지막 달).

    출력: "맞음" | "앞이 빔" | "끝이 이름" | "다른 회사 의심" | "가격 없음"
    """
    if p_first is None or pd.isna(p_first):
        return "가격 없음"
    p_first, p_last = pd.Timestamp(p_first), pd.Timestamp(p_last)
    month_start = first - pd.offsets.MonthBegin(1)
    if p_first > last or p_last < month_start:
        return "다른 회사 의심"
    if p_first > max(month_start, pd.Timestamp("2011-06-01")) + pd.Timedelta(days=tol_days):
        return "앞이 빔"
    if p_last < min(last, SEAL) - pd.Timedelta(days=tol_days + 31):
        return "끝이 이름"
    return "맞음"


def split_adjust(raw: pd.DataFrame) -> pd.DataFrame:
    """Tiingo raw_close·volume·split_factor → 분할 반영 close·volume (마지막 날 기준, 순수 함수)."""
    f = raw["split_factor"].fillna(1.0).replace(0, 1.0).astype(float)
    cum = f.cumprod()
    k = cum / cum.iloc[-1]
    return pd.DataFrame({"close": raw["raw_close"] * k, "volume": raw["volume"] / k}, index=raw.index)


def earnings_dates(ev: pd.DataFrame) -> pd.DataFrame:
    """한 회사의 이벤트(pick_events 결과) → 분기마다 실적 발표 1건 (순수 함수).

    10-Q·10-K마다 (분기말, 그 제출일] 사이 첫 8-K 2.02를 실적 발표로 본다. 없으면 10-Q·10-K 자체를 발표로 본다.
    나머지 8-K 2.02(월별 실적 등)는 버린다.
    출력 열: report_date, filing_date, accn, source('8-K 2.02'|'10-Q'|'10-K'), q_filing_date
    """
    q = ev[ev["kind"].isin(["10-Q", "10-K"])].sort_values("filing_date")
    k = ev[ev["kind"] == "8-K 2.02"].sort_values(["filing_date", "accn"])
    rows = []
    for _, r in q.iterrows():
        rd = r["report_date"] if isinstance(r["report_date"], str) else ""
        if not rd:
            continue
        m = k[(k["filing_date"] > rd) & (k["filing_date"] <= r["filing_date"])]
        src = m.iloc[0] if len(m) else r
        rows.append({"report_date": rd, "filing_date": src["filing_date"], "accn": src["accn"],
                     "source": "8-K 2.02" if len(m) else r["kind"], "q_filing_date": r["filing_date"]})
    out = pd.DataFrame(rows, columns=["report_date", "filing_date", "accn", "source", "q_filing_date"])
    return out.drop_duplicates("report_date")


# ── 입출력 ──────────────────────────────────────────────────────────────────


def _wiki_get(params: dict) -> dict:
    import requests

    for attempt in range(5):
        r = requests.get(WIKI_API, params={**params, "format": "json", "formatversion": 2},
                         headers={"User-Agent": "nasdaq100-screener/1.0 (research)"}, timeout=30)
        if r.status_code != 429:
            break
        wait = int(r.headers.get("Retry-After", 0) or 0) or 10 * (attempt + 1)
        print(f"[wiki] 429 — {wait}초 대기 후 다시 ({attempt + 1}/5)", flush=True)
        time.sleep(wait)
    r.raise_for_status()
    return r.json()


def month_ends() -> list[pd.Timestamp]:
    return [p.to_timestamp(how="end").normalize() for p in pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M")]


def sp500_on(d: pd.Timestamp, hist: pd.DataFrame) -> set[str]:
    """fja05680 S&P 500 명단에서 d 당일 또는 그 전 마지막 행."""
    h = hist[hist.index <= d]
    return set(h.iloc[-1]["tickers"].split(",")) if len(h) else set()


def step_sp100() -> None:
    """위키 옛 판 → 매달 말 S&P 100 명단, S&P 500 명단과 대조."""
    WIKI_CACHE.mkdir(parents=True, exist_ok=True)
    revs, cont = [], {}
    while True:
        r = _wiki_get({"action": "query", "prop": "revisions", "titles": "S&P 100", "rvlimit": 500,
                       "rvstart": "2022-01-01T00:00:00Z", "rvend": "2010-01-01T00:00:00Z", "rvdir": "older",
                       "rvprop": "ids|timestamp", **cont})
        revs += r["query"]["pages"][0]["revisions"]
        if "continue" not in r:
            break
        cont = r["continue"]
    rv = pd.DataFrame(revs)
    rv["ts"] = pd.to_datetime(rv["timestamp"]).dt.tz_localize(None)
    rv = rv.sort_values("ts")
    hist = pd.read_csv(SP500_HIST, index_col=0, parse_dates=True)
    rows, prev = [], None
    for me in month_ends():
        cut = me + pd.Timedelta(days=1)
        r = rv[rv["ts"] < cut].iloc[-1]
        path = WIKI_CACHE / f"{r['revid']}.txt"
        if not path.exists():
            j = _wiki_get({"action": "query", "prop": "revisions", "revids": int(r["revid"]),
                           "rvprop": "content", "rvslots": "main"})
            path.write_text(j["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"], encoding="utf-8")
            time.sleep(1.5)
        tick = parse_sp100_wikitext(path.read_text(encoding="utf-8"))
        sp5 = {t.replace(".", "-") for t in sp500_on(me, hist)}
        missing = sorted(tick - sp5)
        rows.append({"month_end": me.date(), "revid": r["revid"], "rev_date": r["ts"].date(),
                     "rev_age_days": (me - r["ts"].normalize()).days, "n": len(tick),
                     "not_in_sp500": ";".join(missing), "added": ";".join(sorted(tick - prev)) if prev else "",
                     "removed": ";".join(sorted(prev - tick)) if prev else "", "tickers": ";".join(sorted(tick))})
        prev = tick
    out = pd.DataFrame(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE / "sp100_monthly.csv", index=False, encoding="utf-8-sig")
    print(f"[sp100] 옛 판 {len(rv)}개 · 달 {len(out)} · 명단 수 {out['n'].min()}~{out['n'].max()}")
    print(f"[sp100] 합집합 {len(set(';'.join(out['tickers']).split(';')))} · 판 나이 중앙값 {out['rev_age_days'].median():.0f}일, 최대 {out['rev_age_days'].max()}일")
    bad = out[out["not_in_sp500"] != ""]
    print(f"[sp100] S&P 500 명단에 없는 티커가 있는 달 {len(bad)}/{len(out)}")
    print(bad[["month_end", "rev_date", "not_in_sp500"]].to_string(index=False))


def ndx_members() -> dict[pd.Timestamp, set[str]]:
    """매달 말 나스닥 100 시점별 명단 (data.universe_history, E1과 같은 방법)."""
    from data import universe_history as uh

    cur = set(pd.read_csv(ROOT / "data" / "universe_fallback.csv", comment="#")["ticker"])
    return {me: set(uh.point_in_time_universe(me.date(), cur)) for me in month_ends()}


def sp100_members() -> dict[pd.Timestamp, set[str]]:
    d = pd.read_csv(CACHE / "sp100_monthly.csv", parse_dates=["month_end"])
    return {r["month_end"]: set(str(r["tickers"]).split(";")) for _, r in d.iterrows()}


def member_periods(m: dict[pd.Timestamp, set[str]]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """티커 → (처음 달 말, 마지막 달 말)."""
    out: dict[str, list] = {}
    for me, ts in sorted(m.items()):
        for t in ts:
            out.setdefault(t, [me, me])[1] = me
    return {t: (a, b) for t, (a, b) in out.items()}


def expected_names() -> dict[str, set[str]]:
    """티커 → 기대 회사 이름들: 나스닥 100 변경 표 + S&P 100 옛 판 명단 표 + S&P 500 점검 identity.csv."""
    out: dict[str, set[str]] = {}

    def add(d):
        for t, ns in d.items():
            out.setdefault(t, set()).update(ns)

    if NDX_WIKITEXT.exists():
        add(parse_ticker_names(NDX_WIKITEXT.read_text(encoding="utf-8")))
    for f in sorted(WIKI_CACHE.glob("*.txt")):
        add(parse_ticker_names(f.read_text(encoding="utf-8")))
    if SP500_IDENTITY.exists():
        i = pd.read_csv(SP500_IDENTITY, index_col=0)
        add({t: {n} for t, n in i["expected_name"].dropna().items()})
    return out


SEARCH_FAILED: list[str] = []


def name_search(name: str, ua: str) -> list[tuple[int, str]]:
    """EDGAR 회사 이름 검색(efts 자동완성, S&P 500 점검과 같은 방법). 출력 [(cik, 이름)] 상위 10. 캐시."""
    import requests

    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:80]
    path = SEARCH_CACHE / f"{key}.json"
    if path.exists():
        return [(int(c), n) for c, n in json.loads(path.read_text(encoding="utf-8"))]
    d = None
    for q in (name, re.sub(r"[^A-Za-z0-9 ]+", " ", name).strip()):
        r = requests.get("https://efts.sec.gov/LATEST/search-index", params={"keysTyped": q},
                         headers={"User-Agent": ua}, timeout=30)
        time.sleep(0.12)
        if r.status_code == 200:
            d = r.json()
            break
    if d is None:
        SEARCH_FAILED.append(name)  # 실패는 모아 보고 (캐시하지 않음)
        return []
    hits = [(int(h["_id"]), h["_source"]["entity"]) for h in d.get("hits", {}).get("hits", [])][:10]
    SEARCH_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hits), encoding="utf-8")
    return hits


def fetch_submissions(cik: int, ua: str) -> list[dict]:
    """submissions 본문 + 추가 파일. 캐시 data/cache/edgar/submissions/."""
    import requests

    SUB_CACHE.mkdir(parents=True, exist_ok=True)
    out = []
    names = [f"CIK{cik:010d}.json"]
    while names:
        name = names.pop(0)
        path = SUB_CACHE / name
        if path.exists():
            j = json.loads(path.read_text(encoding="utf-8"))
        else:
            r = requests.get(SUB_URL.format(name=name), headers={"User-Agent": ua}, timeout=60)
            time.sleep(0.12)
            r.raise_for_status()
            j = r.json()
            path.write_text(json.dumps(j), encoding="utf-8")
        out.append(j)
        if "filings" in j:
            # 2011-10 이후와 겹치는 추가 파일만
            names += [f["name"] for f in j["filings"].get("files", []) if f.get("filingTo", "9999") >= "2011-10-01"]
    return out


def step_events() -> None:
    """두 명단 합집합의 CIK·제출 기록 → events.csv, cik.csv. 실패 종목은 모아서 출력."""
    from dotenv import load_dotenv

    from data import edgar

    load_dotenv(ROOT / ".env")
    ua = edgar.get_user_agent()
    print(f"[events] SEC_USER_AGENT 있음: {bool(ua)}")
    ndx, sp = member_periods(ndx_members()), member_periods(sp100_members())
    tickers = sorted(set(ndx) | set(sp))
    cmap = edgar.fetch_ticker_to_cik()
    ident = pd.read_csv(SP500_IDENTITY, index_col=0) if SP500_IDENTITY.exists() else pd.DataFrame()
    rn = renames()
    names = expected_names()
    searched: dict[str, tuple[float, str]] = {}
    rows, ev, failed = [], [], []
    for i, t in enumerate(tickers):
        a = min(p[0] for p in (ndx.get(t), sp.get(t)) if p)
        b = max(p[1] for p in (ndx.get(t), sp.get(t)) if p)
        cands = []
        if t in ident.index and not pd.isna(ident.loc[t, "cik"]):
            cands.append((int(ident.loc[t, "cik"]), "S&P 500 점검 identity.csv"))
        c = cmap.get(t.upper().replace("-", "."), cmap.get(t.upper()))
        if c is not None and all(c != x for x, _ in cands):
            cands.append((int(c), "SEC 티커 목록(현재)"))
        new = rn.get(t)
        c = cmap.get(new.replace("-", ".")) if new else None
        if c is not None and all(c != x for x, _ in cands):
            cands.append((int(c), f"SEC 티커 목록(바뀐 티커 {new})"))
        pick = None
        for cik, src in cands:
            try:
                ev_rows = pick_events(submissions_rows(fetch_submissions(cik, ua)), "2011-07-01", "2022-04-30")
            except Exception as exc:  # noqa: BLE001 — 실패는 모아서 보고
                failed.append((t, f"{src} CIK {cik} 받기 실패: {type(exc).__name__}"))
                continue
            q = ev_rows[ev_rows["kind"].isin(["10-Q", "10-K"])]
            # 기간 앞뒤 120일 여유 (편입 2~3달 종목도 10-Q·10-K 2건 이상 잡히게)
            n_in = ((q["filing_date"] >= (a - pd.Timedelta(days=150)).strftime("%Y-%m-%d"))
                    & (q["filing_date"] <= (b + pd.Timedelta(days=120)).strftime("%Y-%m-%d"))).sum()
            if n_in >= 2:  # 명단 기간에 10-Q·10-K가 2건 이상이어야 같은 회사로 본다
                pick = (cik, src, ev_rows)
                break
        if pick is None and t in names and t not in NAME_SEARCH_SKIP:
            # 이름 검색 (S&P 500 점검과 같은 확인: 명단 기간 10-Q·10-K 2건 이상 + 이름 유사도 0.6 이상)
            best = None
            for nm in sorted(names[t]):
                for cik, _ in name_search(nm, ua):
                    if any(cik == x for x, _ in cands):
                        continue
                    try:
                        js = fetch_submissions(cik, ua)
                    except Exception:  # noqa: BLE001 — 후보 하나 실패는 건너뜀
                        continue
                    er = pick_events(submissions_rows(js), "2011-07-01", "2022-04-30")
                    q = er[er["kind"].isin(["10-Q", "10-K"])]
                    n_in = ((q["filing_date"] >= (a - pd.Timedelta(days=150)).strftime("%Y-%m-%d"))
                            & (q["filing_date"] <= (b + pd.Timedelta(days=120)).strftime("%Y-%m-%d"))).sum()
                    ed = [js[0].get("name", "")] + [f.get("name", "") for f in js[0].get("formerNames", [])]
                    sim = max(name_sim(nm, x) for x in ed)
                    if n_in >= 2 and sim >= NAME_SIM_MIN and (best is None or sim > best[3]):
                        best = (cik, f"이름 검색({nm})", er, sim, js[0].get("name", ""))
            if best:
                pick = best[:3]
                searched[t] = (best[3], best[4])
        if pick is None:
            if not cands and t not in names:
                failed.append((t, "CIK 없음(기대 이름도 없음)"))
            elif not cands:
                failed.append((t, f"CIK 없음({NAME_SEARCH_SKIP.get(t, '이름 검색도 실패')})"))
            elif not any(f[0] == t for f in failed):
                failed.append((t, "명단 기간에 10-Q·10-K 없음(외국 기업이거나 티커 재사용)"))
            rows.append({"ticker": t, "cik": None, "cik_source": "", "first": a.date(), "last": b.date(),
                         "in_ndx": t in ndx, "in_sp100": t in sp})
            continue
        cik, src, er = pick
        rows.append({"ticker": t, "cik": cik, "cik_source": src, "first": a.date(), "last": b.date(),
                     "in_ndx": t in ndx, "in_sp100": t in sp,
                     "name_sim": searched[t][0] if t in searched else None,
                     "edgar_name": searched[t][1] if t in searched else None})
        ev.append(er.assign(ticker=t, cik=cik))
        if (i + 1) % 50 == 0:
            print(f"[events] {i + 1}/{len(tickers)}", flush=True)
    pd.DataFrame(rows).to_csv(CACHE / "cik.csv", index=False, encoding="utf-8-sig")
    allev = pd.concat(ev, ignore_index=True)
    allev["acceptance_et"] = allev["acceptance"].map(acceptance_to_et)
    allev.to_csv(CACHE / "events.csv", index=False, encoding="utf-8-sig")
    print(f"[events] 대상 {len(tickers)} (나스닥100 {len(ndx)} · S&P100 {len(sp)} · 겹침 {len(set(ndx) & set(sp))})")
    print(f"[events] CIK 확정 {sum(r['cik'] is not None for r in rows)} · 실패 {len(failed)}")
    for t, why in failed:
        print(f"  실패 {t}: {why}")
    print(f"[events] 이름 검색으로 찾음 {len(searched)}")
    for t, (sim, nm) in sorted(searched.items()):
        print(f"  {t}: {nm} (유사도 {sim:.2f})")
    if SEARCH_FAILED:
        print(f"[events] 이름 검색 요청 실패 {len(SEARCH_FAILED)}: {SEARCH_FAILED}")
    print(allev["kind"].value_counts().to_string())


def renames() -> dict[str, str]:
    """옛 티커 → 지금 티커 (S&P 500 점검 renames.csv + EXTRA_RENAMES)."""
    out = dict(EXTRA_RENAMES)
    if SP500_RENAMES.exists():
        r = pd.read_csv(SP500_RENAMES)
        out.update(dict(zip(r["old"], r["final_ticker"])))
    return out


def _px_one(t: str) -> pd.DataFrame | None:
    p1, p2 = BT_CACHE / f"{t}.parquet", PRICE_CACHE / f"{t}.csv"
    d = None
    if p1.exists():
        d = pd.read_parquet(p1)[["close", "volume"]]
    elif p2.exists():
        d = pd.read_csv(p2, index_col=0, parse_dates=True)
    if d is None:
        return None
    d = d[d.index <= SEAL].dropna(subset=["close"])
    return d if len(d) else None


def load_px(t: str) -> tuple[pd.DataFrame | None, str]:
    """close·volume 일봉 (2021-12-31까지)과 출처.

    순서: backtest 캐시·yfinance(그 티커) → 바뀐 티커 → S&P 500 점검 Tiingo 캐시(raw_close, 분할 미반영).
    """
    d = _px_one(t)
    if d is not None:
        return d, "yfinance"
    new = renames().get(t)
    if new:
        d = _px_one(new)
        if d is not None:
            return d, f"yfinance({new})"
    # Tiingo: S&P 500 점검 캐시는 그 점검의 최종 판정이 "있음"인 것만, SW1 캐시는 price_check가 통과한 것만
    ok500 = set()
    if SP500_FINAL_GAP.exists():
        g = pd.read_csv(SP500_FINAL_GAP, index_col=0)
        ok500 = set(g.index[(g["status"] == "있음") & (g["price_source"] == "tiingo")])
    oksw = set()
    if PRICE_CHECK.exists():
        c = pd.read_csv(PRICE_CHECK)
        oksw = set(c.loc[c["use"], "ticker"])
    for f, src, ok in ((SP500_TIINGO / f"{t}.csv", "tiingo", t in ok500), (TIINGO_CACHE / f"{t}.csv", "tiingo(SW1)", t in oksw)):
        if not ok or not f.exists():
            continue
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        if len(d) and "raw_close" in d:
            d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
            d = split_adjust(d.sort_index())
            d = d[d.index <= SEAL].dropna(subset=["close"])
            if len(d):
                return d, src
    return None, "없음"


def step_prices() -> None:
    """backtest 캐시에 없는 종목만 yfinance(auto_adjust=False) Close·Volume → sw1 캐시. 실패 목록 출력."""
    import yfinance as yf

    cik = pd.read_csv(CACHE / "cik.csv")
    PRICE_CACHE.mkdir(parents=True, exist_ok=True)
    rn = renames()
    need = [x for t in cik["ticker"] for x in ([t] + ([rn[t]] if t in rn else []))
            if not (BT_CACHE / f"{x}.parquet").exists() and not (PRICE_CACHE / f"{x}.csv").exists()]
    print(f"[prices] 받을 종목 {len(need)}")
    bad = []
    for t in need:
        try:
            h = yf.Ticker(t).history(start="2011-06-01", end="2022-01-01", auto_adjust=False)
        except Exception:  # noqa: BLE001 — 실패 종목은 모아서 보고
            h = pd.DataFrame()
        if h is None or h.empty:
            bad.append(t)
            pd.DataFrame(columns=["close", "volume"]).to_csv(PRICE_CACHE / f"{t}.csv")
            continue
        h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
        pd.DataFrame({"close": h["Close"], "volume": h["Volume"]}).to_csv(PRICE_CACHE / f"{t}.csv")
    print(f"[prices] yfinance 실패 {len(bad)}: {', '.join(bad)}")
    src = {t: load_px(t)[1] for t in cik["ticker"]}
    print(pd.Series(src).value_counts().to_string())
    print("[prices] 최종 없음:", ", ".join(t for t, v in src.items() if v == "없음"))


def shares_median_mcap(cik: int, px: pd.DataFrame) -> float | None:
    """EDGAR 표지 주식 수(dei) × 그날 분할 반영 종가의 중앙값 (USD). 자료 없으면 None."""
    from data import edgar

    f = edgar.fetch_company_facts(int(cik))
    ents = []
    for tax, tag in (("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")):
        ents = [e for e in edgar.extract_fact_entries(f, tax, tag) if e.get("unit") == "shares"]
        if len({e["end"] for e in ents}) >= 4:
            break
    if not ents:
        return None
    df = pd.DataFrame(ents)
    per = df.groupby(["end", "accn"]).agg(val=("val", "sum"), filed=("filed", "max")).reset_index()
    per = per.sort_values("filed").drop_duplicates("end", keep="last")
    sh = pd.Series(per["val"].values, index=pd.to_datetime(per["end"])).sort_index()
    sh = sh[(sh.index >= px.index.min()) & (sh.index <= px.index.max())]
    if sh.empty:
        return None
    # 표지 주식 수는 그날 기준이라 분할 반영 전 원본 가격(raw)과 곱한다 (호출부가 raw를 넘긴다)
    c = px["close"].reindex(sh.index, method="ffill")
    return float((sh * c).median())


def step_tiingo() -> None:
    """CIK는 있는데 가격이 없는 종목만 Tiingo(raw) → 날짜·시가총액 같은 회사 확인 → price_check.csv.

    Tiingo 키는 환경변수 TIINGO_API_KEY (값은 출력하지 않는다). 한도(429)면 멈추고 남은 목록을 보고한다.
    """
    import requests
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = os.environ.get("TIINGO_API_KEY", "").strip()
    print(f"[tiingo] TIINGO_API_KEY 있음: {bool(key)}")
    cik = pd.read_csv(CACHE / "cik.csv").dropna(subset=["cik"])
    TIINGO_CACHE.mkdir(parents=True, exist_ok=True)
    rows, stopped = [], None
    # 이미 판정한 것은 다시 판정하되 (네트워크 없음), 받는 건 캐시에 없을 때만
    prev_use = PRICE_CHECK.exists()
    if prev_use:
        PRICE_CHECK.unlink()  # load_px가 옛 판정을 쓰지 않게
    for _, r in cik.iterrows():
        t = r["ticker"]
        if load_px(t)[0] is not None:
            continue
        path = TIINGO_CACHE / f"{t}.csv"
        if not path.exists() and stopped is None:
            resp = requests.get(f"https://api.tiingo.com/tiingo/daily/{t}/prices",
                                params={"startDate": "2011-06-01", "endDate": "2021-12-31", "token": key, "format": "json"},
                                timeout=30)
            time.sleep(0.5)
            if resp.status_code == 429:
                stopped = t
            elif resp.status_code == 200 and isinstance(resp.json(), list):
                recs = resp.json()
                pd.DataFrame([{"date": x["date"][:10], "raw_close": x["close"], "volume": x["volume"],
                               "split_factor": x.get("splitFactor", 1.0)} for x in recs],
                             columns=["date", "raw_close", "volume", "split_factor"]).to_csv(path, index=False)
            else:
                pd.DataFrame(columns=["date", "raw_close", "volume", "split_factor"]).to_csv(path, index=False)
        row = {"ticker": t, "cik": int(r["cik"]), "first": r["first"], "last": r["last"]}
        if not path.exists():
            row.update(status="한도로 못 받음", use=False)
            rows.append(row)
            continue
        raw = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        if raw.empty:
            row.update(status="Tiingo 자료 없음", use=False)
            rows.append(row)
            continue
        real = raw.index[raw["volume"].fillna(0) > 0]
        p_first, p_last = raw.index.min(), (real[-1] if len(real) else raw.index[-1])
        dc = judge_period(pd.Timestamp(r["first"]), pd.Timestamp(r["last"]), p_first, p_last)
        mc = shares_median_mcap(int(r["cik"]), raw.rename(columns={"raw_close": "close"}))
        use = dc != "다른 회사 의심" and (mc is None or mc >= MCAP_MIN_USD or t in PRICE_KEEP)
        row.update(price_first=p_first.date(), price_last=p_last.date(), date_check=dc,
                   mcap_median_b=round(mc / 1e9, 2) if mc else None, status="판정", use=use)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(PRICE_CHECK, index=False, encoding="utf-8-sig")
    print(f"[tiingo] 대상 {len(out)} · 사용 {int(out['use'].sum()) if len(out) else 0}")
    if len(out):
        print(out.to_string(index=False))
    if stopped:
        print(f"[tiingo] 한도(429)로 {stopped}부터 못 받음 — 다시 실행하면 이어 받는다")


# ── 점검 ────────────────────────────────────────────────────────────────────


def eps_check(sample: list[str], ev: pd.DataFrame, cik: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """표본 종목별: 발표일 수, 10-Q·10-K 수, EPS 전년 같은 분기 대비 계산 가능 비율 (E1 core.e1.quarterly_values)."""
    from core import e1
    from data import edgar

    accepted = json.loads(ACCEPTED_CACHE.read_text(encoding="utf-8")) if ACCEPTED_CACHE.exists() else {}
    out = []
    for t in sample:
        c = cik.set_index("ticker").loc[t]
        row = {"ticker": t, "기간": f"{c['first']}~{c['last']}", "CIK": "있음" if not pd.isna(c["cik"]) else "없음"}
        e = ev[(ev["ticker"] == t) & (ev["filing_date"] >= str(START.date())) & (ev["filing_date"] <= str(SEAL.date()))]
        row["8-K 2.02(전체)"] = int((e["kind"] == "8-K 2.02").sum())
        q = e[e["kind"].isin(["10-Q", "10-K"])].sort_values("filing_date")
        row["10-Q·10-K"] = len(q)
        if pd.isna(c["cik"]) or q.empty:
            out.append(row)
            continue
        facts = edgar.fetch_company_facts(int(c["cik"]))
        ents = edgar.extract_fact_entries(facts, "us-gaap", "EarningsPerShareDiluted")
        sp = None
        ep = E1_PRICE_CACHE / f"{t}.csv"
        if ep.exists():
            pdf = pd.read_csv(ep, index_col=0, parse_dates=True)
            s = pdf["split"].fillna(0.0)
            sp = s[s > 0].astype(float)
        ok_late = ok_now = 0
        k202 = e[e["kind"] == "8-K 2.02"].sort_values("filing_date")
        lag = []
        for _, r in q.iterrows():
            qe = pd.Timestamp(r["report_date"]) if isinstance(r["report_date"], str) and r["report_date"] else None
            if qe is None:
                continue
            vals = e1.quarterly_values(ents, pd.Timestamp(r["filing_date"]), sp)
            has = lambda d: len(vals) and (abs((vals.index - d).days) <= 7).any()  # noqa: E731
            ok = bool(has(qe) and has(qe - pd.Timedelta(days=364)))
            ok_late += ok
            # 이 분기의 8-K 2.02: 분기말 뒤 ~ 이 10-Q·10-K 제출일 사이 마지막 것
            k = k202[(k202["filing_date"] > str(qe.date())) & (k202["filing_date"] <= r["filing_date"])]
            if len(k):
                a_ = k.iloc[-1]
                et = pd.Timestamp(accepted[a_["accn"]]) if a_["accn"] in accepted else acceptance_to_et(a_["acceptance"])
                d0 = reaction_day(et, days)
                lag.append((pd.Timestamp(r["filing_date"]) - pd.Timestamp(k.iloc[-1]["filing_date"])).days)
                ok_now += ok and d0 is not None and pd.Timestamp(r["filing_date"]) <= d0
        row["EPS 증감 가능(10-Q·K 제출 뒤)"] = ok_late
        row["가능 비율"] = round(ok_late / len(q), 2)
        row["반응일에 이미 가능"] = ok_now
        row["8-K→10-Q 지연 중앙(일)"] = float(pd.Series(lag).median()) if lag else None
        out.append(row)
    return pd.DataFrame(out)


def acceptance_spot_check(ev: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """무작위 n건의 acceptance 변환값을 EDGAR 공시 색인 페이지 Accepted와 대조."""
    import requests
    from dotenv import load_dotenv

    from data import edgar

    load_dotenv(ROOT / ".env")
    ua = edgar.get_user_agent()
    path = CACHE / "acceptance_spot.csv"
    if path.exists():
        return pd.read_csv(path)
    s = ev[(ev["filing_date"] >= "2012-01-01") & (ev["filing_date"] <= "2021-12-31")].sample(n, random_state=SAMPLE_SEED)
    rows = []
    for _, r in s.iterrows():
        acc = r["accn"]
        u = f"https://www.sec.gov/Archives/edgar/data/{int(r['cik'])}/{acc.replace('-', '')}/{acc}-index.htm"
        h = requests.get(u, headers={"User-Agent": ua}, timeout=30).text
        time.sleep(0.15)
        m = re.search(r"Accepted</div>\s*<div class=\"info\">([^<]+)", h)
        rows.append({"ticker": r["ticker"], "kind": r["kind"], "filing_date": r["filing_date"], "raw": r["acceptance"],
                     "converted_et": str(acceptance_to_et(r["acceptance"])), "index_et": m.group(1).strip() if m else ""})
    d = pd.DataFrame(rows)
    d["match"] = d["converted_et"] == d["index_et"]
    d.to_csv(path, index=False, encoding="utf-8-sig")
    return d


ACCEPTED_CACHE = CACHE / "accepted.json"


def parse_index_accepted(html: str) -> str | None:
    """EDGAR 공시 색인 페이지 → Accepted 값('YYYY-MM-DD HH:MM:SS', 동부시간)."""
    m = re.search(r'Accepted</div>\s*<div class="info">([^<]+)', html)
    return m.group(1).strip() if m else None


def earnings_table() -> pd.DataFrame:
    """CIK 있는 모든 종목의 실적 발표 (2012~2021 제출, 그 전 달 말 명단에 있던 것만)."""
    ev = pd.read_csv(CACHE / "events.csv", dtype={"report_date": str})
    ndx, sp = ndx_members(), sp100_members()
    rows = []
    for t, g in ev.groupby("ticker"):
        e = earnings_dates(g)
        e = e[(e["filing_date"] >= str(START.date())) & (e["filing_date"] <= str(SEAL.date()))]
        rows.append(e.assign(ticker=t, cik=int(g["cik"].iloc[0])))
    e = pd.concat(rows, ignore_index=True)
    me = (pd.to_datetime(e["filing_date"]) - pd.offsets.MonthEnd(1)).dt.normalize()  # 그 전 달 말 명단 (미래 명단 안 씀)
    e["in_ndx"] = [t in ndx.get(m, set()) for t, m in zip(e["ticker"], me)]
    e["in_sp100"] = [t in sp.get(m, set()) for t, m in zip(e["ticker"], me)]
    return e[e["in_ndx"] | e["in_sp100"]].reset_index(drop=True)


def step_accepted() -> None:
    """실적 발표 건마다 공시 색인 페이지의 Accepted(동부시간)를 받는다. 캐시 accepted.json, 실패는 모아 보고."""
    import requests
    from dotenv import load_dotenv

    from data import edgar

    load_dotenv(ROOT / ".env")
    ua = edgar.get_user_agent()
    e = earnings_table()
    cache = json.loads(ACCEPTED_CACHE.read_text(encoding="utf-8")) if ACCEPTED_CACHE.exists() else {}
    todo = [(c, a) for c, a in zip(e["cik"], e["accn"]) if a not in cache]
    print(f"[accepted] 실적 발표 {len(e)} · 이미 받음 {len(e) - len(todo)} · 받을 것 {len(todo)}", flush=True)
    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock, last = threading.Lock(), [0.0]

    def one(cik: int, acc: str) -> tuple[str, str | None, str]:
        # 동시 6개, 시작 간격 0.125초 이상 (SEC 초당 10회 이하)
        with lock:
            wait = last[0] + 0.125 - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last[0] = time.monotonic()
        u = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{acc}-index.htm"
        try:
            r = requests.get(u, headers={"User-Agent": ua}, timeout=30)
        except Exception as exc:  # noqa: BLE001 — 실패는 모아 보고
            return acc, None, type(exc).__name__
        v = parse_index_accepted(r.text) if r.status_code == 200 else None
        return acc, v, "" if v else f"HTTP {r.status_code}"

    failed = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, (acc, v, why) in enumerate(ex.map(lambda x: one(*x), todo)):
            if v:
                cache[acc] = v
            else:
                failed.append((acc, why))
            if (i + 1) % 500 == 0:
                ACCEPTED_CACHE.write_text(json.dumps(cache), encoding="utf-8")
                print(f"[accepted] {i + 1}/{len(todo)} · 실패 {len(failed)}", flush=True)
    ACCEPTED_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    print(f"[accepted] 완료 · 캐시 {len(cache)} · 실패 {len(failed)}")
    for a, why in failed[:30]:
        print(f"  실패 {a}: {why}")


def step_check() -> None:
    """표본 EPS·acceptance 대조·연도별 신호 건수 출력. 결과 표는 data/cache/sw1/에 저장."""
    pd.set_option("display.width", 250)
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    ev = pd.read_csv(CACHE / "events.csv", parse_dates=["acceptance_et"])
    cik = pd.read_csv(CACHE / "cik.csv")
    qqq = pd.read_parquet(BT_CACHE / "QQQ.parquet")["close"]
    qqq = qqq[qqq.index <= SEAL]
    days = qqq.index

    # 1) acceptance 대조
    spot = acceptance_spot_check(ev)
    print(f"[check] acceptance 변환 대조 {int(spot['match'].sum())}/{len(spot)} 일치")
    print(spot.to_string(index=False))

    # 2) 표본 20 (나스닥 100 합집합에서 고정 시드)
    ndx_t = sorted(cik[cik["in_ndx"]]["ticker"])
    sample = sorted(random.Random(SAMPLE_SEED).sample(ndx_t, SAMPLE_N))
    eps = eps_check(sample, ev, cik, days)
    eps.to_csv(CACHE / "sample20_eps.csv", index=False, encoding="utf-8-sig")
    print(f"[check] 표본 20 (시드 {SAMPLE_SEED})")
    print(eps.to_string(index=False))

    # 3) 실적 발표 (분기마다 1건, 명단에 있던 달만) → 색인 페이지 Accepted 시각 → 반응일·신호
    ndx, sp = ndx_members(), sp100_members()
    k = earnings_table()
    acc = json.loads(ACCEPTED_CACHE.read_text(encoding="utf-8")) if ACCEPTED_CACHE.exists() else {}
    k["accepted_et"] = k["accn"].map(lambda a: pd.Timestamp(acc[a]) if a in acc else None)
    k["session"] = k["accepted_et"].map(lambda x: session_of(x) if x is not None else "시각 없음")
    k["d0"] = k["accepted_et"].map(lambda x: reaction_day(x, days) if x is not None else None)
    k["d0_literal"] = k["filing_date"].map(lambda x: next_trading_day(pd.Timestamp(x), days))
    pxs: dict[str, pd.DataFrame | None] = {}
    res = []
    for _, r in k.iterrows():
        t = r["ticker"]
        if t not in pxs:
            pxs[t] = load_px(t)[0]
        px = pxs[t]
        a = signal_on(px, qqq, r["d0"]) if px is not None and r["d0"] is not None else None
        b = signal_on(px, qqq, r["d0_literal"]) if px is not None and r["d0_literal"] is not None else None
        res.append({"price": px is not None, "judged": a is not None,
                    "signal": bool(a and a["signal"]), "signal_literal": bool(b and b["signal"])})
    k = pd.concat([k.reset_index(drop=True), pd.DataFrame(res)], axis=1)
    k["year"] = pd.to_datetime(k["filing_date"]).dt.year
    k.to_csv(CACHE / "announcements.csv", index=False, encoding="utf-8-sig")

    print("[check] 실적 발표 출처")
    print(k["source"].value_counts().to_string())
    print("[check] 발표 시각 구분 (색인 페이지 Accepted, 동부시간)")
    print(k["session"].value_counts().to_string())
    jx = k[k["session"] != "시각 없음"]
    raw = ev.drop_duplicates("accn").set_index("accn")["acceptance"]
    same = sum(1 for a, y in zip(jx["accn"], jx["accepted_et"]) if a in raw.index and acceptance_to_et(raw[a]) == y)
    print(f"[check] JSON acceptance(두 번 변환 가정)이 색인 값과 같은 비율 {same}/{len(jx)}")
    for name, sel in (("나스닥100", k["in_ndx"]), ("S&P100", k["in_sp100"]), ("합집합", k["in_ndx"] | k["in_sp100"])):
        s_ = k[sel]
        tab = s_.groupby("year").agg(발표=("ticker", "size"), 가격있음=("price", "sum"), 판정=("judged", "sum"),
                                     신호=("signal", "sum"), 신호_다음날=("signal_literal", "sum"))
        tab.loc["합계"] = tab.sum()
        print(f"── {name} ──")
        print(tab.to_string())

    # 3-b) 10-Q 확인판 (참고, 결정 3: 주 규칙에는 넣지 않음). 신호마다 그 분기 10-Q·10-K 제출일 기준 EPS 전년 같은 분기 대비
    from core import e1
    from data import edgar

    sig = k[k["signal"]].copy()
    eps_rows = []
    facts_by: dict[int, list] = {}
    for _, r in sig.iterrows():
        c = int(r["cik"])
        if c not in facts_by:
            try:
                facts_by[c] = edgar.extract_fact_entries(edgar.fetch_company_facts(c), "us-gaap", "EarningsPerShareDiluted")
            except Exception:  # noqa: BLE001 — 실패는 '계산 못 함'으로 센다
                facts_by[c] = []
        vals = e1.quarterly_values(facts_by[c], pd.Timestamp(r["q_filing_date"]), None)
        qe = pd.Timestamp(r["report_date"])
        cur = vals[abs((vals.index - qe).days) <= 7] if len(vals) else vals
        old = vals[abs((vals.index - (qe - pd.Timedelta(days=364))).days) <= 7] if len(vals) else vals
        ok = len(cur) > 0 and len(old) > 0
        eps_rows.append({"eps_ok": ok, "eps_up": bool(ok and cur.iloc[-1] > old.iloc[-1]),
                         "q_lag": (pd.Timestamp(r["q_filing_date"]) - pd.Timestamp(r["d0"])).days})
    sig = pd.concat([sig.reset_index(drop=True), pd.DataFrame(eps_rows)], axis=1)
    sig.to_csv(CACHE / "signals_eps_ref.csv", index=False, encoding="utf-8-sig")
    print("[check] 10-Q 확인판 (참고)")
    for name, sel in (("나스닥100", sig["in_ndx"]), ("S&P100", sig["in_sp100"]), ("합집합", sig["in_ndx"] | sig["in_sp100"])):
        x = sig[sel]
        print(f"  {name}: 신호 {len(x)} · EPS 증감 계산 가능 {int(x['eps_ok'].sum())} · 그중 증가 {int(x['eps_up'].sum())}"
              f" · 반응일→10-Q 지연 중앙 {x['q_lag'].median():.0f}일 · 반응일에 이미 10-Q {int((x['q_lag'] <= 0).sum())}")

    # 4) 명단 대비 덮은 비율 (종목·달 기준): CIK와 가격이 둘 다 있는 비율
    ok = set(cik.dropna(subset=["cik"])["ticker"])
    for name, m in (("나스닥100", ndx), ("S&P100", sp)):
        tot = sum(len(v) for me, v in m.items() if me >= START - pd.Timedelta(days=31))
        has = sum(1 for me, v in m.items() if me >= START - pd.Timedelta(days=31) for t in v
                  if t in ok and (pxs.get(t) is not None if t in pxs else load_px(t)[0] is not None))
        print(f"[check] {name} 종목·달 {tot} 중 CIK·가격 모두 있음 {has} ({has / tot:.1%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["sp100", "events", "prices", "tiingo", "accepted", "check"], required=True)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    {"sp100": step_sp100, "events": step_events, "prices": step_prices, "tiingo": step_tiingo,
     "accepted": step_accepted, "check": step_check}[a.step]()


if __name__ == "__main__":
    main()
