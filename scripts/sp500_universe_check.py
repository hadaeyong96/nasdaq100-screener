"""S&P 500 시점별 구성 종목 무료 확보 가능성 점검 (일회성 조사, 백테스트·결과 계산 없음).

목적: R1(매출 성장)·PG1(PEG)을 아직 안 쓴 종목군(S&P 500 중 나스닥 100이 아닌 종목,
2012~2021)으로 검증할 수 있는지 판단할 재료를 모은다. 결과는
docs/research/sp500_universe_check.md에 사람이 정리한다.

출처(둘 다 무료):
- A: GitHub fja05680/sp500 "S&P 500 Historical Components & Changes (Updated).csv" (MIT)
- B: 위키백과 "List of S&P 500 companies" 2025-05-27 판(oldid=1292523673)의 현재 표 + 변경 표 역산
  (2026년 현재 판에는 변경 표가 빠져 있어 옛 판을 쓴다)

표본 30종목의 EDGAR·가격 확보율은 scripts/e1_experiments.py의 load_edgar·load_prices를
그대로 쓴다. 가격 캐시는 E1 캐시와 섞이지 않게 data/cache/sp500/prices/로 돌린다.
"""

from __future__ import annotations

import random
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CACHE = ROOT / "data" / "cache" / "sp500"
FJA_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
FJA_PATH = CACHE / "fja05680_updated.csv"
WIKI_OLDID = 1292523673
WIKI_URL = f"https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&oldid={WIKI_OLDID}"
WIKI_PATH = CACHE / f"wiki_rev{WIKI_OLDID}.html"
START, END = pd.Timestamp("2011-12-31"), pd.Timestamp("2021-12-31")
CHECK_DATES = [pd.Timestamp("2012-01-31"), pd.Timestamp("2016-01-31"), pd.Timestamp("2021-12-31")]
SAMPLE_N, SAMPLE_SEED = 30, 20261008


def _download(url: str, path: Path) -> None:
    """캐시에 없으면 받는다. 실패는 그대로 올린다."""
    if path.exists():
        return
    import requests

    resp = requests.get(url, headers={"User-Agent": "nasdaq100-screener research"}, timeout=60)
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)


def norm(t: str) -> str:
    """티커를 yfinance 형식으로 (BRK.B → BRK-B)."""
    return t.strip().upper().replace(".", "-")


def load_fja() -> pd.DataFrame:
    """출력: date 인덱스, tickers(set, 원래 표기 — 'AAL-199702'처럼 옛 티커는 -YYYYMM 꼬리)."""
    df = pd.read_csv(FJA_PATH, parse_dates=["date"]).set_index("date").sort_index()
    df["tickers"] = df["tickers"].map(lambda s: set(s.split(",")))
    return df


def fja_on(df: pd.DataFrame, as_of: pd.Timestamp) -> set[str]:
    """as_of 이하 마지막 행의 구성 종목(원래 표기)."""
    return df.loc[:as_of].iloc[-1]["tickers"]


def base(t: str) -> str:
    """'AAL-199702' → 'AAL'. 꼬리가 없으면 그대로."""
    return re.sub(r"-\d{6}$", "", t)


def load_wiki() -> tuple[set[str], pd.DataFrame]:
    """출력: (그 판의 현재 구성 종목, 변경 표 DataFrame[date, added, removed])."""
    tables = pd.read_html(WIKI_PATH)
    cur = {norm(t) for t in tables[0]["Symbol"].astype(str)}
    ch = tables[1]
    ch.columns = ["date", "added", "added_name", "removed", "removed_name", "reason"]
    ch["date"] = pd.to_datetime(ch["date"], format="%B %d, %Y", errors="coerce")
    for c in ("added", "removed"):
        ch[c] = ch[c].map(lambda v: norm(v) if isinstance(v, str) and v.strip() else None)
    return cur, ch[["date", "added", "removed"]]


def wiki_on(cur: set[str], ch: pd.DataFrame, as_of: pd.Timestamp) -> set[str]:
    """현재 구성에서 as_of 이후 변경을 거꾸로 되돌린다 (편입 → 빼고, 편출 → 넣고)."""
    out = set(cur)
    for _, r in ch[ch["date"] > as_of].sort_values("date", ascending=False).iterrows():
        if r["added"]:
            out.discard(r["added"])
        if r["removed"]:
            out.add(r["removed"])
    return out


def n100_on(as_of: pd.Timestamp) -> set[str]:
    """나스닥 100 시점별 구성 (E1과 같은 방식: universe_fallback.csv + 위키 변경 역산 캐시)."""
    from data import universe_history as uh

    cur = set(pd.read_csv(ROOT / "data" / "universe_fallback.csv", comment="#")["ticker"])
    return {norm(t) for t in uh.point_in_time_universe(as_of.date(), cur)}


def main() -> None:
    _download(FJA_URL, FJA_PATH)
    _download(WIKI_URL, WIKI_PATH)
    fja = load_fja()
    wcur, wch = load_wiki()
    months = pd.date_range(START, END, freq="ME")

    print("== 출처 A(fja05680) 범위")
    print(f"행 {len(fja)}, {fja.index.min().date()} ~ {fja.index.max().date()}")
    inwin = fja.loc[START - pd.Timedelta(days=40):END]
    gaps = inwin.index.to_series().diff().dt.days
    print(f"2011-11~2021-12 행 {len(inwin)}, 최대 행 간격 {int(gaps.max())}일 ({gaps.idxmax().date()} 직전)")

    print("== 출처 B(위키 옛 판) 변경 표")
    print(f"현재 표 {len(wcur)}종목, 변경 {len(wch)}행, 날짜 못 읽은 행 {int(wch['date'].isna().sum())}")
    w_in = wch[(wch["date"] > START) & (wch["date"] <= END)]
    print(f"2012-01~2021-12 변경 {len(w_in)}행 (편입 {w_in['added'].notna().sum()}, 편출 {w_in['removed'].notna().sum()})")

    print("== 매월 말 A·B 비교 (티커는 yfinance 형식, A는 꼬리 뗀 이름)")
    rows = []
    for m in months:
        a = {norm(base(t)) for t in fja_on(fja, m)}
        b = wiki_on(wcur, wch, m)
        rows.append({"month": m.date(), "A": len(a), "B": len(b), "A_only": len(a - b), "B_only": len(b - a)})
    cmp = pd.DataFrame(rows)
    print(cmp.describe().loc[["min", "mean", "max"]].round(1).to_string())
    print("A·B 차이 상위 5개월:")
    print(cmp.assign(diff=cmp.A_only + cmp.B_only).nlargest(5, "diff").to_string(index=False))

    print("== 확인 시점 종목 수와 나스닥 100 겹침 (A 기준)")
    for d in CHECK_DATES:
        a_raw = fja_on(fja, d)
        a = {norm(base(t)) for t in a_raw}
        n = n100_on(d)
        print(f"{d.date()}: S&P500 {len(a)} · 나스닥100 {len(n)} · 겹침 {len(a & n)} · S&P만 {len(a - n)} · (B 기준 S&P {len(wiki_on(wcur, wch, d))})")

    print("== 2012-01~2021-12 S&P500 중 나스닥100 아닌 종목 전체 (A, 원래 표기)")
    pool_raw = set()
    for m in months[1:]:
        n = n100_on(m)
        pool_raw |= {t for t in fja_on(fja, m) if norm(base(t)) not in n}
    tailed = sorted(t for t in pool_raw if base(t) != t)
    cur_now = {norm(t) for t in fja.iloc[-1]["tickers"]}
    left = sorted(t for t in pool_raw if base(t) == t and norm(t) not in cur_now)
    print(f"전체 {len(pool_raw)} · 옛 티커 꼬리(편출·이름 변경) {len(tailed)} · 꼬리 없이 지금 명단에 없음 {len(left)}")

    print("== 표본 30종목 EDGAR·가격 (E1 load_edgar·load_prices 재사용)")
    rnd = random.Random(SAMPLE_SEED)
    sample_raw = sorted(rnd.sample(sorted(pool_raw), SAMPLE_N))
    sample = [norm(base(t)) for t in sample_raw]
    print("표본:", ", ".join(f"{r}" for r in sample_raw))
    import e1_experiments as e1
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # SEC_USER_AGENT (값은 출력하지 않는다)
    e1.PRICE_CACHE = CACHE / "prices"
    facts, _, failed, foreign = e1.load_edgar(sample)
    prices, missing = e1.load_prices(sample)
    rev_tags = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet")
    from data import edgar

    out = []
    for raw, t in zip(sample_raw, sample):
        f = facts.get(t)
        rev_q = 0
        if f is not None:
            ents = [e for tag in rev_tags for e in edgar.extract_fact_entries(f, "us-gaap", tag)]
            rev_q = len({e.get("end") for e in ents if e.get("form") == "10-Q" and "2011-06" <= str(e.get("end", "")) <= "2021-12-31"})
        fail = next((why for x, why in failed if x == t), "")
        px = prices.get(t)
        out.append({
            "원래 표기": raw, "티커": t,
            "EDGAR": "성공" if f is not None else ("외국/10-Q 없음" if t in foreign else f"실패: {fail}"),
            "매출 10-Q 분기 수": rev_q,
            "가격": "있음" if px is not None else "없음",
            "가격 첫날": px.index.min().date() if px is not None else "",
            "가격 끝날": px.index.max().date() if px is not None else "",
        })
    tab = pd.DataFrame(out)
    print(tab.to_string(index=False))
    tailed_s = tab["원래 표기"] != tab["티커"]
    print(f"EDGAR 성공 {int((tab['EDGAR'] == '성공').sum())}/{SAMPLE_N} · 가격 있음 {int((tab['가격'] == '있음').sum())}/{SAMPLE_N}")
    print(f"옛 티커 표본 {int(tailed_s.sum())}: EDGAR 성공 {int((tab[tailed_s]['EDGAR'] == '성공').sum())}, 가격 {int((tab[tailed_s]['가격'] == '있음').sum())}")
    print(f"현행 티커 표본 {int((~tailed_s).sum())}: EDGAR 성공 {int((tab[~tailed_s]['EDGAR'] == '성공').sum())}, 가격 {int((tab[~tailed_s]['가격'] == '있음').sum())}")
    tab.to_csv(CACHE / "sample30.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
