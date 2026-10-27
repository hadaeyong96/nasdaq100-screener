"""실적 동반 시총 성장 전략(mcap-growth) 검증 구역 ③ 1단계 — 데이터 확인만 (수익 계산 없음, docs/validation_zones.md 2장 4번).

    python scripts/mcap_growth_check.py gics    # 위키 「List of S&P 500 companies」 옛 판(분기 첫 거래일 전 마지막 판) → 시점별 GICS 섹터
    python scripts/mcap_growth_check.py facts   # EDGAR companyfacts(가치평가 때 받은 캐시) → 주식 수·매출 사실, 접수일 ≤ 2021-12-31만 저장
    python scripts/mcap_growth_check.py check   # 분기별 확보율 표(① 주식 수 ② 매출 ③ 섹터 ④ 가격) — 수익·점수 분포는 보지 않는다

가드:
- 명단·가격은 scripts/value_score_zone3_data.Data(구역 ③ 종목만, 2011-06-01 ~ 2021-12-31 줄만, 넘는 줄에서 멈춤)로만 읽는다.
- EDGAR 사실은 접수일 ≤ 2021-12-31만 꺼내 저장하고, 저장 뒤 다시 확인해 넘는 줄이 있으면 멈춘다(2022년 이후 사실은 버린 줄 수만 센다).
- 위키 옛 판은 2022-01-01 전 판만 받는다(사실상 2021-10-01 전 판).
- EDGAR 요청(캐시에 없을 때만)은 data.edgar(초당 10회 이하). SEC_USER_AGENT 값은 출력하지 않는다.

출력: data/cache/mcap_growth/ (git 제외) + 화면 요약
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rs26_zone3_check as z  # noqa: E402
import value_score_zone3_check as c  # noqa: E402
from core import mcap_growth as mg  # noqa: E402

OUT = ROOT / "data" / "cache" / "mcap_growth"
WIKI = OUT / "wiki_sp500"
GICS = OUT / "gics_quarterly.csv"
FACTS = OUT / "facts.parquet"
COVER = OUT / "coverage.csv"
FILED_LAST = "2021-12-31"
WIKI_API = "https://en.wikipedia.org/w/api.php"
PAGE = "List of S&P 500 companies"
FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A")
SECTORS = {"Energy", "Materials", "Industrials", "Consumer Discretionary", "Consumer Staples", "Health Care", "Financials",
           "Information Technology", "Telecommunication Services", "Telecommunications Services", "Communication Services",
           "Utilities", "Real Estate"}
SECTOR_NORM = {"Telecommunications Services": "Telecommunication Services"}
MIN_COVER = 0.70


def quarter_starts() -> list[pd.Timestamp]:
    """2012-01 ~ 2021-10 분기 첫 NYSE 거래일(신호일)."""
    import value_score_zone3_data as vd

    days = vd.nyse_days()
    s = pd.Series(days, index=days)
    first = s.groupby([days.year, days.month]).min()
    return [d for d in first if d.month in (1, 4, 7, 10) and pd.Timestamp("2012-01-01") <= d <= pd.Timestamp("2021-12-31")]


# ── GICS (위키 옛 판) ────────────────────────────────────────────────────────


def _wiki_get(params: dict) -> dict:
    import requests

    for attempt in range(5):
        r = requests.get(WIKI_API, params={**params, "format": "json", "formatversion": 2},
                         headers={"User-Agent": "nasdaq100-screener/1.0 (research)"}, timeout=30)
        if r.status_code != 429:
            break
        time.sleep(int(r.headers.get("Retry-After", 0) or 0) or 10 * (attempt + 1))
    r.raise_for_status()
    return r.json()


def _clean(cell: str) -> str:
    """위키 칸 → 글자만: [[a|b]] → b, [url text] → text, {{X|T}} → T, 주석·태그 제거."""
    s = re.sub(r"<ref[^>]*/>|<ref.*?</ref>|<!--.*?-->", "", cell, flags=re.S)
    s = re.sub(r"\{\{(?:[^{}|]*\|)*([^{}|]*)\}\}", r"\1", s)
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[\S+\s+([^\]]*)\]", r"\1", s)
    s = re.sub(r"\[\S+\]", "", s)
    s = re.sub(r"<[^>]+>|'''?", "", s)
    if "|" in s and not s.strip().startswith("|"):  # style="…" | 값
        s = s.split("|")[-1]
    return s.strip()


def parse_sp500_wikitext(text: str) -> dict[str, str]:
    """첫 번째 표(구성 종목)에서 {티커: GICS 섹터}. 칸 중 섹터 이름과 정확히 같은 첫 칸을 섹터로 본다."""
    i = text.find("{|")
    j = text.find("|}", i)
    table = text[i:j]
    out = {}
    for row in table.split("|-")[1:]:
        cells = []
        for line in row.strip("\n").split("\n"):
            line = line.strip()
            if not line.startswith("|") or line.startswith("|+"):
                continue
            cells += [x for x in line[1:].split("||")]
        cells = [_clean(x) for x in cells]
        if len(cells) < 4:
            continue
        tk = z.norm(cells[0].split()[0]) if cells[0] else ""
        sec = next((x for x in cells[1:] if x in SECTORS), None)
        if tk and sec and re.fullmatch(r"[A-Z][A-Z0-9\-]*", tk):
            out[tk] = SECTOR_NORM.get(sec, sec)
    return out


def step_gics() -> None:
    WIKI.mkdir(parents=True, exist_ok=True)
    revs, cont = [], {}
    while True:
        r = _wiki_get({"action": "query", "prop": "revisions", "titles": PAGE, "rvlimit": 500,
                       "rvstart": "2022-01-01T00:00:00Z", "rvend": "2010-01-01T00:00:00Z", "rvdir": "older",
                       "rvprop": "ids|timestamp", **cont})
        revs += r["query"]["pages"][0]["revisions"]
        if "continue" not in r:
            break
        cont = r["continue"]
    rv = pd.DataFrame(revs)
    rv["ts"] = pd.to_datetime(rv["timestamp"]).dt.tz_localize(None)
    rv = rv.sort_values("ts")
    rows, summ = [], []
    for q in quarter_starts():
        r = rv[rv["ts"] < q].iloc[-1]
        path = WIKI / f"{r['revid']}.txt"
        if not path.exists():
            j = _wiki_get({"action": "query", "prop": "revisions", "revids": int(r["revid"]), "rvprop": "content", "rvslots": "main"})
            path.write_text(j["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"], encoding="utf-8")
            time.sleep(1.5)
        m = parse_sp500_wikitext(path.read_text(encoding="utf-8"))
        summ.append({"quarter": q.date(), "revid": r["revid"], "rev_date": r["ts"].date(), "n": len(m),
                     "sectors": len(set(m.values()))})
        rows += [{"quarter": q.date(), "ticker": t, "sector": s} for t, s in m.items()]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(GICS, index=False, encoding="utf-8-sig")
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "gics_revisions.csv", index=False, encoding="utf-8-sig")
    print(f"[gics] 옛 판 {len(rv)}개 중 분기 {len(s)}개 사용 · 판당 종목 {s['n'].min()}~{s['n'].max()} · 섹터 수 {s['sectors'].min()}~{s['sectors'].max()}")
    print(s.to_string(index=False))


# ── EDGAR 사실 ───────────────────────────────────────────────────────────────


def step_facts() -> None:
    from data import edgar

    tickers, _, _ = c.zone_tickers()
    meta = json.loads(c.META.read_text(encoding="utf-8"))
    rows, failed, late, fetched = [], [], 0, 0
    for t in tickers:
        for cik in meta.get(t, {}).get("ciks", []):
            cached = (edgar.FACTS_CACHE_DIR / f"CIK{cik:010d}.json").exists()
            try:
                cf = edgar.fetch_company_facts(cik)
            except Exception as exc:  # noqa: BLE001 — 실패 종목은 모아서 보고
                failed.append(f"{t}/{cik}({type(exc).__name__})")
                continue
            fetched += 0 if cached else 1
            for tax, tags in (("dei", [mg.SHARES_DEI]), ("us-gaap", [mg.SHARES_WA] + mg.REV_TAGS)):
                for tag in tags:
                    for e in edgar.extract_fact_entries(cf, tax, tag):
                        if e.get("form") not in FORMS or not e.get("filed") or not e.get("end"):
                            continue
                        if e["filed"] > FILED_LAST:
                            late += 1
                            continue  # 2022년 이후 접수는 저장하지 않는다
                        rows.append({"ticker": t, "cik": cik, "tag": tag, "start": e.get("start"), "end": e["end"],
                                     "val": float(e["val"]), "filed": e["filed"], "form": e["form"], "accn": e.get("accn"),
                                     "unit": e["unit"]})
    df = pd.DataFrame(rows)
    if (df["filed"] > FILED_LAST).any():
        raise z.ZoneGuardError("2022년 이후 접수 사실이 섞였습니다")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FACTS, index=False)
    print(f"[facts] 종목 {len(tickers)} · 새로 받은 companyfacts {fetched} · 사실 {len(df):,}줄 · 2022년 이후 접수라 버린 사실 {late:,}")
    for tag in [mg.SHARES_DEI, mg.SHARES_WA]:
        print(f"[facts] {tag}: 종목 {df.loc[df['tag'] == tag, 'ticker'].nunique()}")
    print(f"[facts] 매출 태그: 종목 {df.loc[df['tag'].isin(mg.REV_TAGS), 'ticker'].nunique()}")
    print(f"[facts] 실패 {len(failed)} {failed}")


def load_facts() -> dict[str, pd.DataFrame]:
    f = pd.read_parquet(FACTS)
    if (f["filed"] > FILED_LAST).any():
        raise z.ZoneGuardError("사실 캐시에 2022년 이후 접수 사실이 있습니다")
    return {t: g for t, g in f.groupby("ticker")}


def load_gics() -> dict[pd.Timestamp, dict[str, str]]:
    g = pd.read_csv(GICS, encoding="utf-8-sig", parse_dates=["quarter"])
    return {q: dict(zip(x["ticker"], x["sector"])) for q, x in g.groupby("quarter")}


SIC_DIVISIONS = [(100, 999, "SIC A 농림어업"), (1000, 1499, "SIC B 광업"), (1500, 1799, "SIC C 건설"), (2000, 3999, "SIC D 제조"),
                 (4000, 4999, "SIC E 운수·통신·공익"), (5000, 5199, "SIC F 도매"), (5200, 5999, "SIC G 소매"),
                 (6000, 6799, "SIC H 금융·보험·부동산"), (7000, 8999, "SIC I 서비스"), (9100, 9999, "SIC J 공공")]


def sic_division(sic: str | None) -> str | None:
    if not sic:
        return None
    v = int(sic)
    return next((name for a, b, name in SIC_DIVISIONS if a <= v <= b), None)


# ── 확보율 ───────────────────────────────────────────────────────────────────


class Inputs:
    """분기 신호일마다 종목별 입력(주식 수 3시점·매출 성장·섹터·가격 3시점). 수익은 계산하지 않는다."""

    def __init__(self, data=None):
        import value_score_zone3_data as vd

        self.data = data or vd.Data()
        self.days = vd.nyse_days()
        self.di = {d: i for i, d in enumerate(self.days)}
        self.facts = load_facts()
        self.gics = load_gics()
        self.qs = quarter_starts()

    def members(self, q: pd.Timestamp) -> tuple[set[str], set[str]]:
        """신호일 직전 월말 명단 → (가격 파일 있는 종목, 전체). 2012-01-03은 직전 월말 명단(2011-12)이 재사용 명단에 없어
        2012-01 월말 명단을 쓴다(확보율 표시용 — 그 분기는 252거래일 전 가격이 없어 어차피 판정 불가)."""
        prior = [d for d in self.data.mes if d < q]
        me = max(prior) if prior else min(self.data.mes)
        return self.data.mem[me], self.data.mem_all[me]

    def back(self, q, n):
        i = self.di[q] - n
        return self.days[i] if i >= 0 else None

    def sector(self, q, t) -> tuple[str | None, str]:
        """그 분기 위키 판의 GICS → 없으면 그 전 분기들 중 가장 가까운 판의 GICS → 없으면 SIC 대분류."""
        for qq in sorted((x for x in self.gics if x <= q), reverse=True):
            s = self.gics[qq].get(t)
            if s:
                return s, "GICS" if qq == q else "GICS(전 판)"
        d = sic_division(self.data.sic.get(t))
        return (d, "SIC") if d else (None, "")

    def row(self, q, t, skip=21, look=252) -> dict:
        f = self.facts.get(t)
        sh = f[f["tag"].isin([mg.SHARES_DEI, mg.SHARES_WA])] if f is not None else None
        rv = f[f["tag"].isin(mg.REV_TAGS) & (f["unit"] == "USD")] if f is not None else None
        sp = self.data.splits.get(t)
        out = {"ticker": t}
        for k, n in (("t", 0), ("s", skip), ("l", look)):
            d = self.back(q, n)
            px = self.data.price_on(t, d) if d is not None else None
            shr = mg.shares_asof(sh, d, sp) if d is not None else None
            out[f"px_{k}"], out[f"sh_{k}"] = px, shr
            out[f"mcap_{k}"] = px * shr if (px is not None and shr is not None) else None
        # 희석: 12개월 주식 수 증가율 — 두 값을 모두 신호일 기준 분할로 맞춘다
        sh_l_now = out["sh_l"] * mg.split_factor(sp, self.back(q, look), q) if out["sh_l"] is not None and self.back(q, look) is not None else None
        out["share_g"] = out["sh_t"] / sh_l_now - 1 if (out["sh_t"] is not None and sh_l_now) else None
        out["rev_g"] = mg.revenue_growth(rv, q)
        out["sector"], out["sector_src"] = self.sector(q, t)
        return out


def step_check() -> None:
    inp = Inputs()
    rows = []
    for q in inp.qs:
        priced, allm = inp.members(q)
        rs = [inp.row(q, t) for t in sorted(priced)]
        n = len(allm)
        ok_sh = [r for r in rs if r["sh_t"] is not None and r["sh_s"] is not None and r["sh_l"] is not None]
        ok_rev = [r for r in rs if r["rev_g"] is not None]
        ok_sec = [r for r in rs if r["sector"]]
        ok_gics = [r for r in rs if r["sector_src"].startswith("GICS")]
        ok_px = [r for r in rs if r["px_t"] is not None and r["px_s"] is not None and r["px_l"] is not None]
        all4 = [r for r in rs if r["sh_t"] is not None and r["sh_s"] is not None and r["sh_l"] is not None and r["rev_g"] is not None
                and r["sector"] and r["px_t"] is not None and r["px_s"] is not None and r["px_l"] is not None]
        rows.append({"quarter": q.date(), "members": n, "price_file": len(priced),
                     "shares": len(ok_sh) / n, "revenue": len(ok_rev) / n, "sector": len(ok_sec) / n, "gics": len(ok_gics) / n,
                     "price": len(ok_px) / n, "all4": len(all4) / n, "n_all4": len(all4),
                     "share_dei": sum(1 for r in rs if r["sh_t"] is not None) / n})
    m = pd.DataFrame(rows)
    m["judgeable"] = m["all4"] >= MIN_COVER
    m.to_csv(COVER, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    show = m.copy()
    for k in ("shares", "revenue", "sector", "gics", "price", "all4"):
        show[k] = (show[k] * 100).round(1)
    print(show[["quarter", "members", "price_file", "shares", "revenue", "sector", "gics", "price", "all4", "n_all4", "judgeable"]].to_string(index=False))
    print(f"[판정 불가 분기] {[str(d) for d in m.loc[~m['judgeable'], 'quarter']]}")


if __name__ == "__main__":
    for st in (sys.stdout, sys.stderr):
        st.reconfigure(encoding="utf-8")
    step = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"gics": step_gics, "facts": step_facts, "check": step_check}[step]()
