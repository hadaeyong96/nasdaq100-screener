"""신기술 레이더 G0-보완 실행 (docs/research/radar_g0b_rules.md, 결과 보기 전 고정) — 파일·네트워크. 계산은 core.radar.

    python -m data.radar.g0b_build register   # 급증 + 꾸준한 성장 등록 (640,497 후보)
    python -m data.radar.g0b_build papers     # 등록 단어의 산업 분야 논문 수·2단어/3단어 함께 나온 논문 수 (말뭉치 한 번 더)
    python -m data.radar.g0b_build filter     # ①~③ 거르기 + 감지율(처음 20 / 보류 20) + 단어 표본 50
    python -m data.radar.g0b_build link       # ⑤ EDGAR(새 단어만 검색, 문서는 상한·캐시) → 사업 설명 부분 2회 이상 → 연결 + 표본 50

G0 결과(data/cache/radar)를 그대로 쓰고, 새 결과는 data/cache/radar/g0b/에 쓴다. 가격·수익 캐시는 열지 않는다(data.radar.guard).
"""

from __future__ import annotations

import gzip
import json
import sys
import time
import urllib.parse as up
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.radar import edgar_link as el  # noqa: E402
from core.radar import g0b  # noqa: E402
from core.radar import term_monthly as tm  # noqa: E402
from data.radar import edgar_fetch as ef  # noqa: E402
from data.radar import guard  # noqa: E402
from data.radar import term_build as tb  # noqa: E402

OUT = tb.OUT
G = OUT / "g0b"
SEED_SAMPLE = 20261019
HOLDOUT = ["machine translation", "speech recognition", "object detection", "face recognition", "recommender system",
           "knowledge graph", "smart contract", "digital twin", "massive mimo", "millimeter wave", "wireless power transfer",
           "visible light communication", "quantum key distribution", "quantum error correction", "neuromorphic computing",
           "metal organic framework", "unmanned aerial vehicle", "point cloud", "semantic segmentation", "natural language processing"]
RULES = ("burst", "growth")
RULE_KR = {"burst": "급증", "growth": "꾸준한 성장"}


def _grid_totals():
    total = pd.read_csv(OUT / "month_totals.csv", index_col=0, encoding="utf-8-sig")["total"]
    return list(total.index), total.to_numpy(dtype=float)


def step_register() -> None:
    guard.install()
    G.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    terms_tab = pd.read_parquet(OUT / "terms.parquet")
    keep = terms_tab[~terms_tab["boilerplate"]].reset_index(drop=True)
    tid = pd.Series(np.arange(len(keep)), index=keep["term"])
    grid, totv = _grid_totals()
    mid = {m: i for i, m in enumerate(grid)}
    years = sorted(p.stem for p in tb.CORPUS.glob("*.parquet"))
    parts = []
    for y in years:
        d = pd.read_parquet(OUT / "pass2" / f"{y}.parquet")
        i = d["term"].map(tid)
        ok = i.notna().to_numpy()
        parts.append((i.to_numpy()[ok].astype(np.int64), d["month"].map(mid).to_numpy()[ok].astype(np.int64),
                      d["n_docs"].to_numpy()[ok].astype(np.int64)))
    ti = np.concatenate([p[0] for p in parts]); mi = np.concatenate([p[1] for p in parts]); nv = np.concatenate([p[2] for p in parts])
    order = np.argsort(ti, kind="stable")
    ti, mi, nv = ti[order], mi[order], nv[order]
    burst = np.full(len(keep), -1); growth = np.full(len(keep), -1)
    CH = 100_000
    bounds = np.searchsorted(ti, np.arange(0, len(keep) + CH, CH))
    for k in range(len(bounds) - 1):
        lo, n = k * CH, min(CH, len(keep) - k * CH)
        if n <= 0:
            break
        mat = np.zeros((n, len(grid)))
        a, b = bounds[k], bounds[k + 1]
        mat[ti[a:b] - lo, mi[a:b]] = nv[a:b]
        burst[lo:lo + n] = tm.register(mat, totv)
        growth[lo:lo + n] = g0b.register_growth(mat, totv)
    out = pd.DataFrame({"term": keep["term"], "total_docs": keep["total_docs"], "first_month": keep["first_month"],
                        "burst_month": [grid[i] if i >= 0 else None for i in burst],
                        "growth_month": [grid[i] if i >= 0 else None for i in growth]})
    reg = out[out["burst_month"].notna() | out["growth_month"].notna()]
    reg.to_parquet(G / "registered_raw.parquet", index=False)
    same = reg["burst_month"].notna().sum()
    old = int(terms_tab["reg_month"].notna().sum())
    run = tb.log_run("g0b_register", burst=int(same), growth=int(reg["growth_month"].notna().sum()), union=len(reg),
                     g0_burst=old, seconds=round(time.time() - t0))
    print(run)


_CAND: dict = {}


def _init_papers(cands: list[str], pairs: list[tuple[str, str]]) -> None:
    guard.install()
    _CAND["set"] = frozenset(cands)
    _CAND["pairs"] = pairs
    by_b: dict[str, list[str]] = {}
    for b, t in pairs:
        by_b.setdefault(b, []).append(t)
    _CAND["by_b"] = by_b


def _papers_year(year: str) -> str:
    cand, by_b = _CAND["set"], _CAND["by_b"]
    g = pd.read_parquet(tb.CORPUS / f"{year}.parquet", columns=["month", "primary", "title", "abstract"])
    ind, pair = Counter(), Counter()
    for m, p, t, a in zip(g["month"], g["primary"], g["title"].fillna(""), g["abstract"].fillna("")):
        terms = tm.doc_terms(t, a) & cand
        if not terms:
            continue
        if g0b.is_industrial(p):
            for x in terms:
                ind[(x, m)] += 1
        for b in terms:
            for tri in by_b.get(b, ()):
                if tri in terms:
                    pair[(b, tri, m)] += 1
    pd.DataFrame([(x, m, n) for (x, m), n in ind.items()], columns=["term", "month", "n_ind"]).to_parquet(
        G / "papers" / f"{year}_ind.parquet", index=False)
    pd.DataFrame([(b, t, m, n) for (b, t, m), n in pair.items()], columns=["bigram", "trigram", "month", "n_pair"]).to_parquet(
        G / "papers" / f"{year}_pair.parquet", index=False)
    return year


def step_papers() -> None:
    guard.install()
    t0 = time.time()
    reg = pd.read_parquet(G / "registered_raw.parquet")
    cands = sorted(reg["term"])
    tri = [t for t in cands if len(t.split(" ")) == 3]
    bi = [t for t in cands if len(t.split(" ")) == 2]
    pairs = [(b, t) for t in tri for b in (" ".join(t.split(" ")[:2]), " ".join(t.split(" ")[1:])) if b in set(bi)]
    (G / "papers").mkdir(parents=True, exist_ok=True)
    years = sorted(p.stem for p in tb.CORPUS.glob("*.parquet"))
    with ProcessPoolExecutor(max_workers=6, initializer=_init_papers, initargs=(cands, pairs)) as ex:
        list(ex.map(_papers_year, years))
    print(tb.log_run("g0b_papers", candidates=len(cands), pairs=len(pairs), seconds=round(time.time() - t0)))


def _cum_upto(df: pd.DataFrame, keys: list[str], col: str, month_col: str, upto: dict) -> pd.Series:
    """키별 month ≤ upto[키] 합."""
    d = df.copy()
    d["_u"] = [upto.get(k) for k in (d[keys[0]] if len(keys) == 1 else list(zip(*[d[k] for k in keys])))]
    d = d[d["_u"].notna() & (d[month_col] <= d["_u"])]
    return d.groupby(keys)[col].sum()


def step_filter() -> None:
    guard.install()
    reg = pd.read_parquet(G / "registered_raw.parquet")
    years = sorted(p.stem for p in tb.CORPUS.glob("*.parquet"))
    ind = pd.concat([pd.read_parquet(G / "papers" / f"{y}_ind.parquet") for y in years], ignore_index=True)
    pair = pd.concat([pd.read_parquet(G / "papers" / f"{y}_pair.parquet") for y in years], ignore_index=True)
    cset = set(reg["term"])
    tmo = pd.concat([pd.read_parquet(f, columns=["term", "month", "mentions"]) for f in sorted((OUT / "term_monthly").glob("*.parquet"))])
    tmo = tmo[tmo["term"].isin(cset)]
    first_reg = {t: min(x for x in (b, g) if x) for t, b, g in zip(reg["term"], reg["burst_month"], reg["growth_month"])}
    rows = []
    for rule in RULES:
        col = f"{rule}_month"
        r = reg[reg[col].notna()]
        upto = dict(zip(r["term"], r[col]))
        n_all = _cum_upto(tmo, ["term"], "mentions", "month", upto)
        n_ind = _cum_upto(ind, ["term"], "n_ind", "month", upto)
        # ② (가) 조각: B를 품은 등록 3단어 T, 기간 = 두 등록 달 중 늦은 달(T는 두 규칙 중 이른 등록)
        frag = {}
        pr = pair[pair["bigram"].isin(set(r["term"])) & pair["trigram"].isin(set(first_reg))]
        if len(pr):
            ends = {(b, t): max(upto[b], first_reg[t]) for b, t in set(zip(pr["bigram"], pr["trigram"]))}
            both = _cum_upto(pr, ["bigram", "trigram"], "n_pair", "month", ends)
            bt = tmo[tmo["term"].isin(set(pr["bigram"]))]
            for (b, t), n in both.items():
                denom = int(bt.loc[(bt["term"] == b) & (bt["month"] <= ends[(b, t)]), "mentions"].sum())
                if denom and n / denom > g0b.FRAGMENT_SHARE:
                    frag.setdefault(b, []).append(t)
        for t, m in upto.items():
            share = float(n_ind.get(t, 0) / n_all.get(t, np.nan)) if n_all.get(t, 0) else np.nan
            f_field = not (share >= g0b.MIN_INDUSTRIAL_SHARE)
            f_frag = t in frag
            f_gen = g0b.generic_edge(t)
            f_evt = g0b.event_term(t)
            rows.append({"term": t, "rule": rule, "rule_kr": RULE_KR[rule], "reg_month": m, "industrial_share": share,
                         "drop_field": f_field, "drop_fragment": f_frag, "fragment_of": ";".join(frag.get(t, [])),
                         "drop_generic": f_gen, "drop_event": f_evt, "pass": not (f_field or f_frag or f_gen or f_evt)})
    res = pd.DataFrame(rows).merge(reg[["term", "total_docs", "first_month"]], on="term")
    res.to_parquet(G / "registered_filtered.parquet", index=False)
    ok = res[res["pass"]]
    by_year = ok.assign(y=ok["reg_month"].str[:4]).groupby(["rule_kr", "y"]).size().unstack(0, fill_value=0)
    drops = {r: {k: int(res[(res["rule"] == r) & res[k]].shape[0]) for k in ("drop_field", "drop_fragment", "drop_generic", "drop_event")}
             for r in RULES}
    # 감지율 (처음 등장 달 = G0 단어 표의 첫 언급 달, 한 단어는 G0 참고 표)
    first_all = pd.read_parquet(OUT / "terms.parquet", columns=["term", "first_month"]).set_index("term")["first_month"].to_dict()
    first_all.update(pd.read_csv(OUT / "oneword_ref.csv", encoding="utf-8-sig").set_index("term")["first_month"].to_dict())
    det = []
    for group, lst in (("처음 20", tb.TARGETS), ("보류 20", HOLDOUT)):
        for target in lst:
            k = tm.normalize_term(target)
            row = {"group": group, "target": target, "key": k}
            row["first_month"] = first_all.get(k)
            for rule in RULES:
                x = res[(res["term"] == k) & (res["rule"] == rule)]
                if " " not in k:
                    row[f"{rule}"] = "한 단어(구조상 불가)"
                elif x.empty:
                    row[f"{rule}"] = "미등록"
                else:
                    x = x.iloc[0]
                    why = [n for n, f in (("①분야", x.drop_field), ("②가 조각", x.drop_fragment), ("②나 일반어", x.drop_generic), ("③사건", x.drop_event)) if f]
                    row[f"{rule}"] = f"{x.reg_month} 등록" + ("" if x["pass"] else f" → 제외({'·'.join(why)}, 산업 비율 {x.industrial_share:.0%})")
                    row[f"{rule}_pass"] = bool(x["pass"])
                    row[f"{rule}_month"] = x.reg_month
            det.append(row)
    dd = pd.DataFrame(det)
    dd.to_csv(G / "detection.csv", index=False, encoding="utf-8-sig")
    union = ok.sort_values("reg_month").drop_duplicates("term")
    rng = np.random.default_rng(SEED_SAMPLE)
    s = union.iloc[sorted(rng.choice(len(union), size=min(50, len(union)), replace=False))]
    s = s.merge(ok.pivot_table(index="term", columns="rule_kr", values="reg_month", aggfunc="first").reset_index(), on="term", how="left")
    cols = ["term"] + [c for c in ("급증", "꾸준한 성장") if c in s.columns] + ["first_month", "total_docs", "industrial_share"]
    s = s[cols].assign(**{"분류(기술/조각/학문/상투어/사건)": ""})
    s.to_csv(ROOT / "docs" / "research" / "radar_g0b_terms_sample50.csv", index=False, encoding="utf-8-sig")
    stats = {"raw": {r: int(reg[f"{r}_month"].notna().sum()) for r in RULES}, "raw_union": len(reg),
             "pass": {r: int(ok[ok["rule"] == r].shape[0]) for r in RULES}, "pass_union": int(ok["term"].nunique()),
             "drops": drops, "by_year": {c: {k: int(v) for k, v in by_year[c].items()} for c in by_year.columns},
             "detected": {g: {r: int(dd[(dd.group == g)].get(f"{r}_pass", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
                              for r in RULES} for g in ("처음 20", "보류 20")},
             "industrial_share_pass_median": float(ok["industrial_share"].median())}
    (G / "stats_filter.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(dd.to_string(index=False))


def step_link() -> None:
    guard.install()
    ua = ef.edgar.get_user_agent()
    t0 = time.time()
    res = pd.read_parquet(G / "registered_filtered.parquet")
    ok = res[res["pass"] & (res["reg_month"] >= "2001-01") & (res["reg_month"] <= "2021-12")]
    terms = sorted(set(ok["term"]))
    ef.HITS.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for term in terms:  # G0에서 검색 안 한 단어만 새로 검색 (같은 캐시 키)
        f = ef.HITS / f"{ef._key(term)}.parquet"
        if f.exists():
            continue
        rows, frm, total = [], 0, None
        while True:
            r = ef._get_retry("https://efts.sec.gov/LATEST/search-index?" + up.urlencode(el.efts_params(term, frm)), ua)
            hits, total = el.parse_hits(r.json())
            rows += hits
            frm += 100
            if not hits or frm >= min(total, ef.ES_CAP):
                break
        df = pd.DataFrame(rows, columns=["adsh", "filename", "cik", "file_date", "form", "name"])
        df["term"], df["efts_total"] = term, total
        df.to_parquet(f, index=False)
        n_new += 1
    h = pd.concat([pd.read_parquet(ef.HITS / f"{ef._key(t)}.parquet") for t in terms], ignore_index=True)
    h = h[(h["file_date"] >= el.START) & (h["file_date"] <= el.END)]
    iv = ef.intervals()
    by_cik = {c: g for c, g in iv.groupby("cik")}
    h["ticker"] = [el.match_ticker(by_cik[c], c, d) if c in by_cik else None for c, d in zip(h["cik"], h["file_date"])]
    pick = ef.chosen_docs(h)
    fetched_new, failed = 0, []
    for adsh, fn, cik in pick.drop_duplicates(["adsh", "filename"])[["adsh", "filename", "cik"]].itertuples(index=False):
        p = ef._doc_path(adsh, fn)
        if p.exists():
            continue
        try:
            raw = ef._get_retry(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{adsh.replace('-', '')}/{fn}", ua).text
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(gzip.compress(el.html_to_text(raw).encode("utf-8")))
            fetched_new += 1
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{adsh}/{fn}({str(exc)[:50]})")
    rows = []
    cache: dict = {}
    for r in pick.itertuples(index=False):
        p = ef._doc_path(r.adsh, r.filename)
        if not p.exists():
            rows.append({**r._asdict(), "section_found": None, "section_mentions": None, "excerpt": None})
            continue
        if p not in cache:
            cache = {p: gzip.decompress(p.read_bytes()).decode("utf-8")}
        sec = g0b.section(cache[p], r.form)
        n, ex = el.mentions(sec, r.term) if sec else (None, None)
        rows.append({**r._asdict(), "section_found": sec is not None, "section_mentions": n, "excerpt": ex})
    ev = pd.DataFrame(rows)
    ev["linked"] = ev["section_mentions"].fillna(0) >= 2
    ev["review"] = "미검수"
    ev.to_parquet(G / "ticker_term_evidence_g0b.parquet", index=False)
    linked = ev[ev["linked"]]
    rng = np.random.default_rng(SEED_SAMPLE)
    s = linked.iloc[sorted(rng.choice(len(linked), size=min(50, len(linked)), replace=False))].copy() if len(linked) else linked
    s["url"] = [f"https://www.sec.gov/Archives/edgar/data/{int(c)}/{a.replace('-', '')}/{f}" for c, a, f in zip(s["cik"], s["adsh"], s["filename"])]
    s = s.rename(columns={"file_date": "filed", "form": "filing_type", "section_mentions": "mentions"})
    s["분류(개발자/공급자/고객/경쟁/피해/무관)"] = ""
    s[["ticker", "term", "filed", "filing_type", "mentions", "excerpt", "url", "분류(개발자/공급자/고객/경쟁/피해/무관)"]].to_csv(
        ROOT / "docs" / "research" / "radar_g0b_link_sample50.csv", index=False, encoding="utf-8-sig")
    stats = {"terms": len(terms), "new_searched": n_new, "hits": len(h), "matched": int(h["ticker"].notna().sum()),
             "match_pct": float(h["ticker"].notna().mean() * 100) if len(h) else None, "docs_chosen": len(pick),
             "docs_new": fetched_new, "docs_failed": failed, "section_found": int((ev["section_found"] == True).sum()),  # noqa: E712
             "section_missing": int((ev["section_found"] == False).sum()), "linked_rows": int(ev["linked"].sum()),  # noqa: E712
             "linked_tickers": int(linked["ticker"].nunique()), "linked_terms": int(linked["term"].nunique()),
             "seconds": round(time.time() - t0)}
    h.to_parquet(G / "hits_g0b.parquet", index=False)
    (G / "stats_link.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    for s_ in (sys.stdout, sys.stderr):
        s_.reconfigure(encoding="utf-8")
    {"register": step_register, "papers": step_papers, "filter": step_filter, "link": step_link}[sys.argv[1]]()
