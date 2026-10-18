"""신기술 레이더 G0 — 달별 단어 표·급증 등록·감지율 (파일 입출력, 계산은 core.radar.term_monthly).

    python -m data.radar.term_build corpus     # parsed 쪽 → 연도별 말뭉치(중복 id 제거, v1 ≤ 2021-12-31)
    python -m data.radar.term_build terms      # 1차 근사 세기 → 2차 정확 세기 → term_monthly·등록·상투어
    python -m data.radar.term_build check      # 재현성(한 달 따로 다시 세기) + 감지율 표 + 표본 50

입력·출력은 data/cache/arxiv, data/cache/radar(git 제외)만 쓴다. 가격·수익 캐시는 열지 않는다(data.radar.guard).
공개 저장소에는 제목·요약 원문을 커밋하지 않는다 — docs/에는 표현(2~3단어)과 숫자만 쓴다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.radar import term_monthly as tm  # noqa: E402
from data.radar import guard  # noqa: E402

ARX = ROOT / "data" / "cache" / "arxiv"
# RADAR_OUT: 시험 실행용 다른 출력 폴더(작업자 프로세스도 같은 값을 물려받는다). 없으면 data/cache/radar
OUT = Path(os.environ["RADAR_OUT"]) if os.environ.get("RADAR_OUT") else ROOT / "data" / "cache" / "radar"
CORPUS = OUT / "corpus"
WIDTH = 26
MIN_TOTAL = 20
LAST_MONTH = "2021-12"
FIRST_V1 = "1991-01-01"  # 지시 기간 1991~2021 (arXiv 레코드에 v1 날짜가 1991 전인 것이 있다 — 가장 이른 것 1988-11-11)
ONE_WORD = frozenset({"blockchain", "bitcoin", "graphene", "lidar"})  # 감지율 참고 칸 전용 (사용자 결정 2026-10-18)
TARGETS = ["deep learning", "convolutional neural network", "generative adversarial network", "reinforcement learning",
           "large language model", "federated learning", "internet of things", "cloud computing", "edge computing",
           "software defined networking", "blockchain", "bitcoin", "quantum computing", "augmented reality",
           "autonomous driving", "lidar", "solid state battery", "perovskite solar cell", "graphene", "additive manufacturing"]
SAMPLE_SEED = 20261018


def code_version() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    d = subprocess.run(["git", "status", "--porcelain", "--", "core/radar", "data/radar"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() + ("+dirty" if d.stdout.strip() else "")


def log_run(step: str, **kw) -> dict:
    run = {"run_id": uuid.uuid4().hex[:12], "step": step, "at": datetime.now(timezone.utc).isoformat(),
           "code_version": code_version(), **kw}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(run, ensure_ascii=False, default=str) + "\n")
    return run


def step_corpus() -> None:
    guard.install()
    st = json.loads((ARX / "state.json").read_text(encoding="utf-8"))
    if not st.get("done"):
        raise SystemExit("arXiv 수집이 아직 끝나지 않았습니다 (state.json done 아님)")
    pages = sorted((ARX / "parsed").glob("page_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in pages], ignore_index=True)
    n_raw = len(df)
    df = df.drop_duplicates("id", keep="first")
    bad_date = df["v1_date"].str.match(r"^\d{4}-\d{2}-\d{2}$") == False  # noqa: E712
    df = df[~bad_date & (df["v1_date"] >= FIRST_V1) & (df["v1_date"] <= "2021-12-31")]
    df["month"] = df["v1_date"].str[:7]
    miss = {c: int((df[c].fillna("") == "").sum()) for c in ("title", "abstract", "categories")}
    CORPUS.mkdir(parents=True, exist_ok=True)
    for y, g in df.groupby(df["month"].str[:4]):
        g.sort_values(["month", "id"]).to_parquet(CORPUS / f"{y}.parquet", index=False)
    run = log_run("corpus", pages=len(pages), rows_saved=n_raw, papers=len(df), duplicates=n_raw - len(df) - int(bad_date.sum()),
                  bad_date=int(bad_date.sum()), missing=miss, first_month=df["month"].min(), last_month=df["month"].max())
    print(run)


# ── 1차·2차 세기 (연도별 프로세스) ─────────────────────────────────────────


def _pass1(year: str) -> np.ndarray:
    guard.install()
    g = pd.read_parquet(CORPUS / f"{year}.parquet", columns=["title", "abstract"])
    sk = np.zeros((2, 1 << WIDTH), dtype=np.uint32)
    for t, a in zip(g["title"].fillna(""), g["abstract"].fillna("")):
        tm.sketch_add(sk, list(tm.doc_terms(t, a)), WIDTH)
    return sk


_SK: dict = {}


def _init2(sk_path: str) -> None:
    guard.install()
    _SK["sk"] = np.load(sk_path)


def _pass2(year: str) -> str:
    sk = _SK["sk"]
    g = pd.read_parquet(CORPUS / f"{year}.parquet", columns=["month", "title", "abstract"])
    cnt: dict[str, Counter] = {}
    one: dict[str, Counter] = {}
    tot = g.groupby("month").size().to_dict()
    for m, t, a in zip(g["month"], g["title"].fillna(""), g["abstract"].fillna("")):
        c = cnt.setdefault(m, Counter())
        for term in tm.doc_terms(t, a):
            h0, h1 = tm.hashes(term, WIDTH)
            if sk[0][h0] >= MIN_TOTAL and sk[1][h1] >= MIN_TOTAL:
                c[term] += 1
        o = one.setdefault(m, Counter())
        for w in tm.doc_words(t, a, ONE_WORD):
            o[w] += 1
    rows = [(term, m, n) for m, c in cnt.items() for term, n in c.items()]
    pd.DataFrame(rows, columns=["term", "month", "n_docs"]).to_parquet(OUT / "pass2" / f"{year}.parquet", index=False)
    pd.DataFrame([(w, m, n) for m, c in one.items() for w, n in c.items()], columns=["term", "month", "n_docs"]).to_parquet(
        OUT / "pass2" / f"{year}_oneword.parquet", index=False)
    pd.DataFrame(list(tot.items()), columns=["month", "total"]).to_parquet(OUT / "pass2" / f"{year}_total.parquet", index=False)
    return year


def months_grid(first: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(first, LAST_MONTH, freq="M")]


def step_terms() -> None:
    guard.install()
    t0 = time.time()
    years = sorted(p.stem for p in CORPUS.glob("*.parquet"))
    with ProcessPoolExecutor(max_workers=6, initializer=guard.install) as ex:
        sk = np.zeros((2, 1 << WIDTH), dtype=np.uint32)
        for part in ex.map(_pass1, years):
            sk += part
    np.save(OUT / "sketch.npy", sk)
    print(f"1차 끝 {time.time() - t0:.0f}초", flush=True)
    (OUT / "pass2").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=6, initializer=_init2, initargs=(str(OUT / "sketch.npy"),)) as ex:
        list(ex.map(_pass2, years))
    print(f"2차 끝 {time.time() - t0:.0f}초", flush=True)
    run = log_run("terms")
    rid = run["run_id"]
    total = pd.concat([pd.read_parquet(OUT / "pass2" / f"{y}_total.parquet") for y in years]).set_index("month")["total"]
    grid = months_grid(total.index.min())
    total = total.reindex(grid, fill_value=0)
    totv = total.to_numpy(dtype=float)
    mid = {m: i for i, m in enumerate(grid)}
    # 전체 합(연도별 합을 모아) → 20회 이상만 남김
    sums = pd.concat([pd.read_parquet(OUT / "pass2" / f"{y}.parquet").groupby("term")["n_docs"].sum() for y in years])
    tsum = sums.groupby(level=0).sum()
    terms = np.array(sorted(tsum[tsum >= MIN_TOTAL].index))
    tid = pd.Series(np.arange(len(terms)), index=terms)
    del sums
    parts = []
    for y in years:
        d = pd.read_parquet(OUT / "pass2" / f"{y}.parquet")
        i = d["term"].map(tid)
        ok = i.notna().to_numpy()
        parts.append((i.to_numpy()[ok].astype(np.int64), d["month"].map(mid).to_numpy()[ok].astype(np.int64),
                      d["n_docs"].to_numpy()[ok].astype(np.int64)))
    ti = np.concatenate([p[0] for p in parts])
    mi = np.concatenate([p[1] for p in parts])
    nv = np.concatenate([p[2] for p in parts])
    del parts
    order = np.argsort(ti, kind="stable")
    ti, mi, nv = ti[order], mi[order], nv[order]
    reg = np.full(len(terms), -1)
    boil = np.zeros(len(terms), dtype=bool)
    CH = 100_000
    bounds = np.searchsorted(ti, np.arange(0, len(terms) + CH, CH))
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        lo = k * CH
        n = min(CH, len(terms) - lo)
        if n <= 0:
            break
        mat = np.zeros((n, len(grid)))
        mat[ti[a:b] - lo, mi[a:b]] = nv[a:b]
        boil[lo:lo + n] = tm.boilerplate(mat / np.where(totv > 0, totv, 1))
        reg[lo:lo + n] = tm.register(mat, totv)
    reg_tab = pd.DataFrame({"term": terms, "total_docs": tsum.reindex(terms).to_numpy(), "boilerplate": boil,
                            "reg_month": [grid[i] if i >= 0 and not bl else None for i, bl in zip(reg, boil)],
                            "first_month": [None] * len(terms)})
    first = np.full(len(terms), len(grid))
    np.minimum.at(first, ti, mi)  # 처음 언급된 달
    reg_tab["first_month"] = [grid[i] if i < len(grid) else None for i in first]
    reg_tab.to_parquet(OUT / "terms.parquet", index=False)
    pd.Series(sorted(terms[boil])).to_frame("term").to_csv(OUT / "boilerplate.csv", index=False, encoding="utf-8-sig")
    # term_monthly: 연도별 파일(상투어 제외, 0이 아닌 칸만)
    (OUT / "term_monthly").mkdir(parents=True, exist_ok=True)
    bad = set(terms[boil])
    for y in years:
        d = pd.read_parquet(OUT / "pass2" / f"{y}.parquet")
        d = d[d["term"].isin(tid.index) & ~d["term"].isin(bad)].rename(columns={"n_docs": "mentions"})
        d["total"] = d["month"].map(total).astype(int)
        d["ratio"] = d["mentions"] / d["total"]
        d["run_id"] = rid
        d[["term", "month", "mentions", "total", "ratio", "run_id"]].to_parquet(OUT / "term_monthly" / f"{y}.parquet", index=False)
    total.rename("total").to_frame().to_csv(OUT / "month_totals.csv", encoding="utf-8-sig")
    # 참고: 한 단어 4개 (감지율 전용)
    one = pd.concat([pd.read_parquet(OUT / "pass2" / f"{y}_oneword.parquet") for y in years], ignore_index=True)
    om = one.pivot_table(index="term", columns="month", values="n_docs", aggfunc="sum").reindex(columns=grid).fillna(0.0)
    oreg = tm.register(om.to_numpy(), totv)
    pd.DataFrame({"term": om.index, "reg_month": [grid[i] if i >= 0 else None for i in oreg],
                  "first_month": [grid[int(np.argmax(r > 0))] if (r > 0).any() else None for r in om.to_numpy()]}).to_csv(
        OUT / "oneword_ref.csv", index=False, encoding="utf-8-sig")
    reg_ok = reg_tab[reg_tab["reg_month"].notna()]
    log_run("terms_done", ref_run_id=rid, terms_ge20=len(terms), boilerplate=int(boil.sum()), registered=len(reg_ok),
            seconds=round(time.time() - t0))
    print(f"표현(합 ≥ 20) {len(terms):,} · 상투어 {int(boil.sum())} · 등록 {len(reg_ok):,} · {time.time() - t0:.0f}초")


# ── 점검: 재현성·감지율·표본 ───────────────────────────────────────────────


def recount_month(month: str) -> pd.DataFrame:
    """한 달을 근사 세기 없이 처음부터 다시 센다(재현성 확인용). 출력: term, mentions."""
    g = pd.read_parquet(CORPUS / f"{month[:4]}.parquet", columns=["month", "title", "abstract"])
    g = g[g["month"] == month]
    c = Counter()
    for t, a in zip(g["title"].fillna(""), g["abstract"].fillna("")):
        c.update(tm.doc_terms(t, a))
    return pd.DataFrame(list(c.items()), columns=["term", "mentions"])


def step_check(month: str = "2015-03") -> None:
    guard.install()
    terms = pd.read_parquet(OUT / "terms.parquet")
    tmo = pd.read_parquet(OUT / "term_monthly" / f"{month[:4]}.parquet")
    tmo = tmo[tmo["month"] == month].set_index("term")["mentions"]
    again = recount_month(month).set_index("term")["mentions"]
    keep = set(terms.loc[~terms["boilerplate"], "term"])
    again = again[again.index.isin(keep)]
    same = tmo.sort_index().equals(again.reindex(tmo.index).sort_index()) and set(again.index) == set(tmo.index)
    diff = int((tmo - again.reindex(tmo.index)).abs().sum()) if len(tmo) else 0
    # 감지율 (평가 전용)
    reg = terms.set_index("term")
    one = pd.read_csv(OUT / "oneword_ref.csv", encoding="utf-8-sig").set_index("term")
    rows = []
    for t in TARGETS:
        k = tm.normalize_term(t)
        if " " not in k:
            r = one.loc[k] if k in one.index else None
            rows.append({"target": t, "key": k, "kind": "한 단어(구조상 등록 불가, 참고만)", "registered": False,
                         "reg_month": None, "first_month": None if r is None else r["first_month"],
                         "ref_reg_month": None if r is None or pd.isna(r["reg_month"]) else r["reg_month"]})
            continue
        r = reg.loc[k] if k in reg.index else None
        rows.append({"target": t, "key": k, "kind": "2~3단어",
                     "registered": bool(r is not None and pd.notna(r["reg_month"])),
                     "reg_month": None if r is None else r["reg_month"], "first_month": None if r is None else r["first_month"],
                     "total_docs": None if r is None else int(r["total_docs"]),
                     "boilerplate": None if r is None else bool(r["boilerplate"]), "ref_reg_month": None})
    det = pd.DataFrame(rows)

    def lag(a, b):
        if not a or not b or pd.isna(a) or pd.isna(b):
            return None
        pa, pb = pd.Period(a, "M"), pd.Period(b, "M")
        return (pa - pb).n

    det["lag_months"] = [lag(a, b) for a, b in zip(det["reg_month"], det["first_month"])]
    det["ref_lag_months"] = [lag(a, b) for a, b in zip(det["ref_reg_month"], det["first_month"])]
    det.to_csv(OUT / "detection.csv", index=False, encoding="utf-8-sig")
    ok = terms[terms["reg_month"].notna()].copy()
    ok["year"] = ok["reg_month"].str[:4]
    by_year = ok.groupby("year").size()
    top = (ok.sort_values(["year", "total_docs"], ascending=[True, False]).groupby("year").head(30))
    top[["year", "term", "reg_month", "first_month", "total_docs"]].to_csv(OUT / "registered_top30_by_year.csv", index=False,
                                                                           encoding="utf-8-sig")
    ok[["term", "reg_month", "first_month", "total_docs"]].to_csv(OUT / "registered.csv", index=False, encoding="utf-8-sig")
    rng = np.random.default_rng(SAMPLE_SEED)
    s = ok.iloc[sorted(rng.choice(len(ok), size=min(50, len(ok)), replace=False))]
    s = s.assign(category="")[["term", "reg_month", "first_month", "total_docs", "category"]]
    s.rename(columns={"category": "분류(기술 용어/학술 상투어/기타)"}).to_csv(
        ROOT / "docs" / "research" / "radar_terms_sample50.csv", index=False, encoding="utf-8-sig")
    run = log_run("check", month=month, same=same, abs_diff=diff, n_terms_month=int(len(tmo)), registered=len(ok),
                  detected_of_20=int(det["registered"].sum()))
    print(run)
    print("[연도별 등록]", by_year.to_dict())
    print(det.to_string(index=False))


if __name__ == "__main__":
    for st_ in (sys.stdout, sys.stderr):
        st_.reconfigure(encoding="utf-8")
    {"corpus": step_corpus, "terms": step_terms, "check": step_check}[sys.argv[1]]()
