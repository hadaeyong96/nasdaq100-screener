"""신기술 레이더 G0 — 등록 단어의 10-K·10-Q 회사 연결 (EDGAR 전문 검색, 네트워크·파일).

    python -m data.radar.edgar_fetch hits      # 등록 시점 2001~2021 단어마다 검색 결과(메타) 전부 → 캐시 (이어받기)
    python -m data.radar.edgar_fetch docs      # 티커 매칭된 공시 중 단어×연도 최대 5건 문서 → 언급 수·발췌 (이어받기)
    python -m data.radar.edgar_fetch evidence  # ticker_term_evidence 표 + 검수용 무작위 50건 CSV + 집계

[사용자 결정 2026-10-18] 메타는 전부, 문서는 티커 매칭 공시 중 단어×접수 연도마다 최대 5건(시드 20261018 무작위).
기존 SEC User-Agent(.env SEC_USER_AGENT, 값은 출력하지 않음), 초당 10회 이하(data.edgar 속도 제한).
가격·수익 캐시는 열지 않는다(data.radar.guard). 저장은 data/cache/radar/edgar(git 제외).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
import time
import urllib.parse as up
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.radar import edgar_link as el  # noqa: E402
from data import edgar  # noqa: E402
from data.radar import guard  # noqa: E402

OUT = ROOT / "data" / "cache" / "radar"
ED = OUT / "edgar"
HITS = ED / "hits"
DOCS = ED / "docs"
PER_YEAR = 5
SEED = 20261018
ES_CAP = 10_000  # EDGAR 전문 검색이 넘겨 주는 최대 결과 수
CACHE = ROOT / "data" / "cache"


def _key(term: str) -> str:
    return hashlib.sha1(term.encode("utf-8")).hexdigest()[:16]


def target_terms() -> pd.DataFrame:
    t = pd.read_parquet(OUT / "terms.parquet")
    t = t[t["reg_month"].notna()]
    return t[(t["reg_month"] >= "2001-01") & (t["reg_month"] <= "2021-12")].sort_values("term").reset_index(drop=True)


def intervals() -> pd.DataFrame:
    """시점별 (CIK, 티커, 시작, 끝): S&P 500(identity + fja05680 소속 기간), 나스닥100 ∪ S&P 100(SW1), S&P 400·600(중소형 점검)."""
    rows = []
    idn = pd.read_csv(CACHE / "sp500" / "identity.csv", encoding="utf-8-sig")
    fja = pd.read_csv(CACHE / "sp500" / "fja05680_updated.csv")
    fja = fja[fja["date"] <= "2021-12-31"]
    seen: dict[str, list[str]] = {}
    for d, s in zip(fja["date"], fja["tickers"]):
        for t in s.split(","):
            seen.setdefault(re.sub(r"-\d{6}$", "", t.strip().upper().replace(".", "-")), []).append(d)  # 옛 티커 꼬리 제거
    sw = pd.read_csv(CACHE / "sw1" / "cik.csv", encoding="utf-8-sig")
    sm = pd.read_csv(CACHE / "smallmid" / "tickers.csv", encoding="utf-8-sig")
    # 티커 → CIK (확인된 표 셋: S&P 500 신원표, 나스닥100 ∪ S&P 100, S&P 400·600). S&P 500 소속 기간(fja05680)에 붙인다
    t2c: dict[str, set[int]] = {}
    for t, c in list(zip(idn["ticker"], idn["cik"])) + list(zip(sw["ticker"], sw["cik"])) + list(zip(sm["ticker_hist"], sm["cik"])):
        if pd.notna(c) and pd.notna(t):
            t2c.setdefault(str(t), set()).add(int(c))
    for t, ds in seen.items():
        for c in t2c.get(t, ()):
            rows.append((c, t, min(ds), max(ds)))
    rows += [(int(c), t, f, l) for t, c, f, l in zip(sw["ticker"], sw["cik"], sw["first"], sw["last"]) if pd.notna(c)]
    rows += [(int(c), t, f, l) for c, t, f, l in zip(sm["cik"], sm["ticker_hist"], sm["first"], sm["last"]) if pd.notna(t)]
    return el.build_intervals(rows)


def _get_retry(url: str, ua: str, tries: int = 5):
    """EDGAR 요청 + 일시 오류(HTTP 5xx·네트워크) 재시도 10·20·40·80·160초. 끝내 실패하면 그대로 올린다."""
    for k in range(tries + 1):
        try:
            return edgar._get(url, ua)
        except edgar.EdgarFetchError as exc:
            msg = str(exc)
            if k == tries or not ("HTTP 5" in msg or "네트워크" in msg or "HTTP 429" in msg):
                raise
            wait = 10 * 2 ** k
            print(f"  재시도 {k + 1}/{tries}: {msg.split(' — ')[0]} · {wait}초", flush=True)
            time.sleep(wait)


def step_hits() -> None:
    guard.install()
    ua = edgar.get_user_agent()
    HITS.mkdir(parents=True, exist_ok=True)
    terms = target_terms()
    t0, n_req = time.time(), 0
    for k, term in enumerate(terms["term"]):
        f = HITS / f"{_key(term)}.parquet"
        if f.exists():
            continue
        rows, frm, total = [], 0, None
        while True:
            r = _get_retry("https://efts.sec.gov/LATEST/search-index?" + up.urlencode(el.efts_params(term, frm)), ua)
            n_req += 1
            hits, total = el.parse_hits(r.json())
            rows += hits
            frm += 100
            if not hits or frm >= min(total, ES_CAP):
                break
        df = pd.DataFrame(rows, columns=["adsh", "filename", "cik", "file_date", "form", "name"])
        df["term"], df["efts_total"] = term, total
        df.to_parquet(f, index=False)
        if k % 50 == 0:
            print(f"단어 {k + 1}/{len(terms)} · 요청 {n_req} · {time.time() - t0:.0f}초", flush=True)
    print(f"검색 끝: 단어 {len(terms)} · 요청 {n_req} · {time.time() - t0:.0f}초")


def all_hits() -> pd.DataFrame:
    terms = target_terms()
    parts = [pd.read_parquet(HITS / f"{_key(t)}.parquet") for t in terms["term"] if (HITS / f"{_key(t)}.parquet").exists()]
    h = pd.concat(parts, ignore_index=True)
    h = h[(h["file_date"] >= el.START) & (h["file_date"] <= el.END)]
    iv = intervals()
    by_cik = {c: g for c, g in iv.groupby("cik")}
    h["ticker"] = [el.match_ticker(by_cik[c], c, d) if c in by_cik else None for c, d in zip(h["cik"], h["file_date"])]
    return h


def chosen_docs(h: pd.DataFrame) -> pd.DataFrame:
    m = h[h["ticker"].notna()].copy()
    m["year"] = m["file_date"].str[:4]
    rng = np.random.default_rng(SEED)
    pick = []
    for _, g in m.sort_values(["term", "year", "adsh", "filename"]).groupby(["term", "year"], sort=True):
        idx = rng.choice(len(g), size=min(PER_YEAR, len(g)), replace=False)
        pick.append(g.iloc[sorted(idx)])
    return pd.concat(pick, ignore_index=True) if pick else m.iloc[0:0]


def _doc_path(adsh: str, fn: str) -> Path:
    return DOCS / f"{adsh}_{hashlib.sha1(fn.encode()).hexdigest()[:10]}.txt.gz"


def step_docs() -> None:
    guard.install()
    ua = edgar.get_user_agent()
    DOCS.mkdir(parents=True, exist_ok=True)
    pick = chosen_docs(all_hits()).drop_duplicates(["adsh", "filename"])
    t0, failed = time.time(), []
    for k, (adsh, fn, cik) in enumerate(zip(pick["adsh"], pick["filename"], pick["cik"])):
        p = _doc_path(adsh, fn)
        if p.exists():
            continue
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{adsh.replace('-', '')}/{fn}"
        try:
            raw = _get_retry(url, ua).text
        except Exception as exc:  # noqa: BLE001 — 실패 목록을 모아 보고
            failed.append(f"{adsh}/{fn}({str(exc)[:50]})")
            continue
        p.write_bytes(gzip.compress(el.html_to_text(raw).encode("utf-8")))
        if k % 200 == 0:
            print(f"문서 {k + 1}/{len(pick)} · {time.time() - t0:.0f}초", flush=True)
    (ED / "docs_failed.json").write_text(json.dumps(failed, ensure_ascii=False), encoding="utf-8")
    print(f"문서 끝: {len(pick)}건 · 실패 {len(failed)} · {time.time() - t0:.0f}초")


def step_evidence() -> None:
    guard.install()
    h = all_hits()
    pick = chosen_docs(h)
    keyset = set(zip(pick["term"], pick["adsh"], pick["filename"]))
    ev = h.copy()
    ev["mentions"], ev["excerpt"] = None, None
    texts: dict = {}
    for i, (term, adsh, fn) in enumerate(zip(ev["term"], ev["adsh"], ev["filename"])):
        if (term, adsh, fn) not in keyset:
            continue
        p = _doc_path(adsh, fn)
        if not p.exists():
            continue
        if p not in texts:
            texts = {p: gzip.decompress(p.read_bytes()).decode("utf-8")}
        n, ex = el.mentions(texts[p], term)
        ev.iat[i, ev.columns.get_loc("mentions")] = n
        ev.iat[i, ev.columns.get_loc("excerpt")] = ex
    ev["review"] = "미검수"
    ev = ev.rename(columns={"form": "filing_type", "file_date": "filed"})
    cols = ["ticker", "term", "filed", "filing_type", "mentions", "excerpt", "review", "cik", "adsh", "filename", "name", "efts_total"]
    ev[cols].to_parquet(OUT / "ticker_term_evidence.parquet", index=False)
    fetched = ev[ev["mentions"].notna()]
    rng = np.random.default_rng(SEED)
    s = fetched.iloc[sorted(rng.choice(len(fetched), size=min(50, len(fetched)), replace=False))].copy() if len(fetched) else fetched
    s["url"] = [f"https://www.sec.gov/Archives/edgar/data/{int(c)}/{a.replace('-', '')}/{f}" for c, a, f in zip(s["cik"], s["adsh"], s["filename"])]
    s["분류(개발자/공급자/고객/경쟁/피해/무관)"] = ""
    s[["ticker", "term", "filed", "filing_type", "mentions", "excerpt", "url", "분류(개발자/공급자/고객/경쟁/피해/무관)"]].to_csv(
        ROOT / "docs" / "research" / "radar_link_sample50.csv", index=False, encoding="utf-8-sig")
    terms = target_terms()
    trunc = h.groupby("term")["efts_total"].max()
    stats = {"terms_2001_2021": len(terms), "terms_with_hits": int(h["term"].nunique()), "filings": len(h),
             "unique_filings": int(h[["adsh", "filename"]].drop_duplicates().shape[0]),
             "matched": int(h["ticker"].notna().sum()), "match_fail_pct": float(h["ticker"].isna().mean() * 100) if len(h) else None,
             "matched_tickers": int(h["ticker"].nunique()), "truncated_terms_over_10000": int((trunc >= ES_CAP).sum()),
             "docs_chosen": len(pick), "docs_with_text": int(fetched[["adsh", "filename"]].drop_duplicates().shape[0]),
             "evidence_rows_with_mentions": len(fetched), "zero_mention_docs": int((fetched["mentions"] == 0).sum())}
    (OUT / "edgar_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    for s_ in (sys.stdout, sys.stderr):
        s_.reconfigure(encoding="utf-8")
    {"hits": step_hits, "docs": step_docs, "evidence": step_evidence}[sys.argv[1]]()
