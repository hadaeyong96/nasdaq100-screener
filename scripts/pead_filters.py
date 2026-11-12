"""실적급등 보완 필터 탐색 — SW1 신호 B1·90에 필터 F1~F6 각각 (계획: docs/research/pead_filters_plan.md, 태그 pead-filters-plan).

일회성 리포트 스크립트. 탐색이지 판정이 아니다. 신호·가격·비용·세금은 SW1·타이밍 탐색과 같다(core/sw1, core/pead_timing).
필터 판정은 core/pead_filters(순수 함수), 데이터 받기는 data/pead_filters_data.

실행:
  python -u -m scripts.pead_filters fetch   # 데이터 받기 — 판정 불가 개수만 출력, 수익 계산 안 함
  python -u -m scripts.pead_filters run     # 계산 1회 → docs/research/pead_filters_results.md
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import pead_filters as pf  # noqa: E402
from core import pead_timing as pt  # noqa: E402
from core import sealed_a as sa  # noqa: E402
from core import sw1  # noqa: E402
from data import pead_filters_data as fd  # noqa: E402
from scripts import sealed_a_b1 as b1  # noqa: E402
from scripts import sealed_a_sw1 as sas  # noqa: E402
from scripts import sealed_a_sw1_data as sd  # noqa: E402
from scripts import sw1_data_check as dc  # noqa: E402
from scripts import sw1_experiments as sx  # noqa: E402

PLAN_TAG = "pead-filters-plan"
PLAN = "docs/research/pead_filters_plan.md"
RESULT_MD = ROOT / "docs" / "research" / "pead_filters_results.md"
OUT_JSON = ROOT / "outputs" / "pead_filters.json"
SEED, ITERS, RATE = 20261111, 1000, 0.22
HOLDS = (90, 60)
BASE_KNOWN = {90: (549, 0.67), 60: (601, 0.53)}  # 타이밍 탐색 결과
SEALED_BASE_KNOWN = (336, -0.69)
PERIODS = ((2012, 2016), (2017, 2021))
SEALED_PERIODS = ((2022, 2023), (2024, 2026))
COSTS = sw1.Costs()
NAMES = {"F1": "F1 급등 지킴", "F2": "F2 신고가 근처", "F3": "F3 종목 추세", "F4": "F4 매출 가속", "F5": "F5 이익률 확대",
         "F6": "F6 시장 상승기"}


def verify_plan() -> str:
    """계획 문서 작업 폴더 = 태그 커밋 내용(LF 기준). 다르면 멈춘다. 출력: 태그 커밋 해시(짧게)."""
    work = sx.lf_sha256((ROOT / PLAN).read_bytes())
    tagged = sx.lf_sha256(subprocess.run(["git", "show", f"{PLAN_TAG}:{PLAN}"], cwd=ROOT, check=True,
                                         capture_output=True).stdout)
    if work != tagged:
        raise SystemExit(f"[pead-filters] 계획 문서가 태그 {PLAN_TAG}와 다름 — 멈춤")
    return subprocess.run(["git", "rev-parse", "--short", f"{PLAN_TAG}^{{commit}}"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


# ── 신호 · 필터 판정 ─────────────────────────────────────────────────────────


def main_signals() -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """2012~2021 SW1 신호 636건(재현 확인) · QQQ · 달력."""
    qqq = pd.read_parquet(dc.BT_CACHE / "QQQ.parquet")[["open", "close"]]
    if qqq.index[-1] > sx.SEAL:
        raise SystemExit("[pead-filters] QQQ 봉인일 뒤 행 — 멈춤")
    days = qqq.index
    k = sx.rebuild_announcements(days, qqq["close"])
    ref = pd.read_csv(dc.CACHE / "announcements.csv")
    sig = k[k["signal"]].copy()
    if len(sig) != 636 or set(sig["accn"]) != set(ref.loc[ref["signal"], "accn"]):
        raise SystemExit(f"[pead-filters] 신호 재현 불일치 {len(sig)}건 — 멈춤")
    sig["d0"] = pd.to_datetime(sig["d0"])
    sig = sig[["ticker", "d0", "accn", "cik", "report_date", "q_filing_date", "in_ndx", "in_sp100"]].reset_index(drop=True)
    return sig, qqq, days


def hl_on(kind: str, x: str, t: str, d0: pd.Timestamp, key: str, period: str, fails: list) -> pf.Verdict:
    """F1: 출처별 d0 고가·저가·종가 → 판정. 받기 실패는 fails에 모은다."""
    if kind == "bt":
        h = pd.read_parquet(dc.BT_CACHE / f"{x}.parquet")[["high", "low", "close"]]
    elif kind == "ti":
        h, why = fd.fetch_hl_tiingo(x, key, period)
        if h is None:
            fails.append((t, why))
            return None, "고가·저가 받기 실패"
    else:
        h = fd.fetch_hl_yf(x, period)
        if h is None:
            fails.append((t, "yfinance 빈 응답"))
            return None, "고가·저가 받기 실패"
    if d0 not in h.index:
        return None, "반응일 고가·저가 행 없음"
    r = h.loc[d0]
    return pf.f1_hold(float(r["high"]), float(r["low"]), float(r["close"]))


def xbrl_flags(sig: pd.DataFrame, events: pd.DataFrame, ua: str | None, fails: list) -> pd.DataFrame:
    """F4·F5 판정. 같은 분기 10-Q/10-K Accepted는 캐시(fetch 단계)에서 읽는다. 출력 열 F4, F4_why, F5, F5_why, q_late."""
    acc = fd.q_accepted_cache()
    tags_of: dict[int, dict | None] = {}
    rows = []
    for _, r in sig.iterrows():
        cik, d0 = int(r["cik"]), pd.Timestamp(r["d0"])
        if cik not in tags_of:
            try:
                tags_of[cik] = fd.load_tags(cik, ua)
            except Exception as exc:  # noqa: BLE001 — 실패는 모아 보고
                fails.append((r["ticker"], f"companyfacts {type(exc).__name__}"))
                tags_of[cik] = None
        qa = fd.q_accn_of(events, cik, r["report_date"], r["q_filing_date"])
        qt = pd.Timestamp(acc[qa]) if qa in acc else None
        qf = pd.Timestamp(r["q_filing_date"])
        q_late = not (qf < d0 or (qf == d0 and qt is not None and qt < d0 + pd.Timedelta(hours=pf.CUTOFF_HOUR)))
        tags = tags_of[cik]
        if tags is None:
            v4 = v5 = (None, "companyfacts 받기 실패")
        else:
            kn = {t: pf.known(df, d0, qa, qt) for t, df in tags.items()}
            v4 = pf.f4_rev_accel(kn, pd.Timestamp(r["report_date"]), q_late)
            v5 = pf.f5_margin_up(kn, pd.Timestamp(r["report_date"]), q_late)
        rows.append({"F4": v4[0], "F4_why": v4[1], "F5": v5[0], "F5_why": v5[1], "q_late": q_late, "q_accn": qa})
    return pd.DataFrame(rows, index=sig.index)


def main_flags(sig: pd.DataFrame, qqq: pd.DataFrame, raw: dict, which=pf.FILTERS) -> tuple[pd.DataFrame, list]:
    """2012~2021 신호 → 필터 판정 표(열 F1..F6, Fk_why). 출력: (표, 받기 실패 목록)."""
    key = fd.tiingo_key() if "F1" in which else ""
    fails: list = []
    rows = []
    for _, r in sig.iterrows():
        t, d0 = r["ticker"], pd.Timestamp(r["d0"])
        rec = {}
        if "F1" in which:
            src = sx.ohlc_source(t)
            v = hl_on(src[0], src[1], t, d0, key, "main", fails) if src else (None, "가격 출처 없음")
            rec.update(F1=v[0], F1_why=v[1])
        c = raw[t]["close"] if raw.get(t) is not None else pd.Series(dtype=float)
        for f, fn in (("F2", pf.f2_near_high), ("F3", pf.above_sma)):
            if f in which:
                v = fn(c, d0)
                rec.update({f: v[0], f"{f}_why": v[1]})
        if "F6" in which:
            v = pf.above_sma(qqq["close"], d0)
            rec.update(F6=v[0], F6_why=v[1])
        rows.append(rec)
    out = pd.DataFrame(rows, index=sig.index)
    if "F4" in which or "F5" in which:
        ev = pd.read_csv(dc.CACHE / "events.csv", dtype={"report_date": str})
        out = out.join(xbrl_flags(sig, ev, None, fails))
    return out, fails


# ── 매매 · 지표 ──────────────────────────────────────────────────────────────


class World:
    """한 구간(2012~2021 또는 봉인 ⓐ)의 달력·가격·환율·무작위 후보 명단."""

    def __init__(self, days, al, bench, fx, pool_rows, first_year, n_years, periods):
        self.days, self.al, self.bench, self.fx = days, al, bench, fx
        names = sorted(al)
        self.M = (np.vstack([al[t].open for t in names]), np.vstack([al[t].close for t in names]),
                  np.vstack([al[t].lastv for t in names]))
        self.row_of = {t: i for i, t in enumerate(names)}
        self.pool_rows = pool_rows
        self.first_year, self.n_years, self.periods = first_year, n_years, periods
        self.years = days.year.to_numpy()

    def trades(self, sig: pd.DataFrame, hold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        tr, sk = pt.make_trades(sig, self.al, self.days, "B1", hold)
        return pt.price_trades(tr, self.bench, self.fx, "B1", COSTS, RATE), sk

    def stats(self, sig: pd.DataFrame, hold: int, random: bool) -> dict:
        """신호 부분 집합 → 지표(% 단위). random이면 무작위 1,000회 백분위도."""
        T, sk = self.trades(sig, hold)
        if T.empty:
            return {"n": 0, "mean": np.nan, "median": np.nan, "win": np.nan, "ex_pre": np.nan, "p1": np.nan, "p2": np.nan,
                    "ndx": np.nan, "n_ndx": 0, "sp": np.nan, "n_sp": 0, "pct": np.nan, "skip": {}}
        ex = T["ex_after"] * 100
        per = [float(ex[(T["year"] >= a) & (T["year"] <= b)].mean()) for a, b in self.periods]
        nd, sp = T["in_ndx"].astype(bool), T["in_sp100"].astype(bool)
        S = {"n": int(len(T)), "mean": float(ex.mean()), "median": float(ex.median()), "win": float((ex > 0).mean() * 100),
             "ex_pre": float(T["ex_pre"].mean() * 100), "p1": per[0], "p2": per[1],
             "n_p1": int(((T["year"] >= self.periods[0][0]) & (T["year"] <= self.periods[0][1])).sum()),
             "n_p2": int(((T["year"] >= self.periods[1][0]) & (T["year"] <= self.periods[1][1])).sum()),
             "ndx": float(ex[nd].mean()) if nd.any() else np.nan, "n_ndx": int(nd.sum()),
             "sp": float(ex[sp].mean()) if sp.any() else np.nan, "n_sp": int(sp.sum()), "pct": np.nan,
             "skip": sk["skip"].value_counts().to_dict()}
        if random:
            pools = [self.pool_rows(b, t) for b, t in zip(T["buy_day"], T["ticker"])]
            O, C, L = self.M
            cands = pt.candidates(T, pools, O, C, L, self.bench, self.fx, self.years, "B1", COSTS)
            rm = sx.random_means(cands, ITERS, SEED, RATE, self.first_year, self.n_years)
            S["pct"] = sw1.percentile_of(T["ex_after"].mean(), rm)
            S["r_median"] = float(np.median(rm) * 100)
        return S


def main_world(raw: dict, qqq: pd.DataFrame, days: pd.DatetimeIndex) -> World:
    fx = sx.load_fx(days).to_numpy()
    al = {t: pt.align(v, days) for t, v in raw.items() if v is not None}
    ndx, sp = dc.ndx_members(), dc.sp100_members()
    row_of = {t: i for i, t in enumerate(sorted(al))}

    def pool_rows(b, t):
        me = (pd.Timestamp(b) - pd.offsets.MonthEnd(1)).normalize()
        return np.array(sorted(row_of[x] for x in (ndx.get(me, set()) | sp.get(me, set())) - {t} if x in row_of), dtype=int)

    return World(days, al, pt.align(qqq, days), fx, pool_rows, 2012, 11, PERIODS)


def subset(sig: pd.DataFrame, F: pd.DataFrame, names: tuple[str, ...], want) -> pd.DataFrame:
    """want=True 통과(모든 필터) · False 걸러짐(판정 가능, 하나라도 걸러짐) · None 판정 불가(하나라도)."""
    v = F[list(names)]
    na = v.isna().any(axis=1)
    if want is None:
        return sig[na]
    ok = (v == True).all(axis=1)  # noqa: E712 — object 열(True/False/None)
    return sig[ok] if want else sig[~na & ~ok]


# ── 2022 ~ 2026-09 (봉인 ⓐ) ─────────────────────────────────────────────────


def sealed_setup() -> tuple[pd.DataFrame, World, dict, pd.DataFrame, pd.DataFrame]:
    """봉인 ⓐ 신호(판정·신호, cik·accn 중복 제거)와 World, 종가 dict(cik), QQQ, price_check."""
    b1.verify_preregistration()
    qq = sas.load_bench()
    days = qq.index
    k = sd.classify(days, qq["close"])
    v = k[k["status"] != "기간 밖"]
    prices, pc = sas.load_prices()
    fx, _ = sas.load_fx(days)
    sig = v[(v["status"] == "판정") & v["signal"]].copy()
    sig["d0"] = pd.to_datetime(sig["d0"])
    sig["sym"] = sig["ticker"]
    sig["ticker"] = sig["cik"].astype(int).astype(str)
    sig = sig.drop_duplicates(["ticker", "d0"])[["ticker", "d0", "sym", "accn", "cik", "report_date", "q_filing_date",
                                                 "in_ndx", "in_sp100"]].reset_index(drop=True)
    al = {c: pt.align(p[["open", "close"]], days) for c, p in prices.items()}
    m = sd.members()
    cik_of = {r["ticker"]: str(int(r["cik"])) for _, r in pc[pc["use"]].iterrows()}
    row_of = {t: i for i, t in enumerate(sorted(al))}

    def pool_rows(b, t):
        me = (pd.Timestamp(b) - pd.offsets.MonthEnd(1)).normalize()
        ts = m["ndx"].get(me, set()) | m["sp100"].get(me, set())
        return np.array(sorted(row_of[c] for c in {cik_of[x] for x in ts if x in cik_of} - {t} if c in row_of), dtype=int)

    W = World(days, al, pt.align(qq, days), fx.to_numpy(), pool_rows, 2022, 5, SEALED_PERIODS)
    return sig, W, prices, qq, pc


def sealed_flags(sig: pd.DataFrame, prices: dict, qq: pd.DataFrame, pc: pd.DataFrame, which) -> tuple[pd.DataFrame, list]:
    """봉인 ⓐ 신호의 필터 판정(which만). F1 고가·저가는 price_check의 price_symbol로 yfinance."""
    fails: list = []
    psym = pc.set_index("ticker")["price_symbol"].to_dict()
    rows = []
    for _, r in sig.iterrows():
        d0, rec = pd.Timestamp(r["d0"]), {}
        if "F1" in which:
            v = hl_on("yf", psym.get(r["sym"], r["sym"]), r["sym"], d0, "", "sealed", fails)
            rec.update(F1=v[0], F1_why=v[1])
        c = prices[r["ticker"]]["close"] if r["ticker"] in prices else pd.Series(dtype=float)
        for f, fn in (("F2", pf.f2_near_high), ("F3", pf.above_sma)):
            if f in which:
                v = fn(c, d0)
                rec.update({f: v[0], f"{f}_why": v[1]})
        if "F6" in which:
            v = pf.above_sma(qq["close"], d0)
            rec.update(F6=v[0], F6_why=v[1])
        rows.append(rec)
    out = pd.DataFrame(rows, index=sig.index)
    if "F4" in which or "F5" in which:
        ev = pd.read_csv(sd.CACHE / "events.csv", dtype={"report_date": str})
        out = out.join(xbrl_flags(sig, ev, None, fails))
    return out, fails


# ── 받기 단계 ────────────────────────────────────────────────────────────────


def q_todo(sig: pd.DataFrame, events: pd.DataFrame) -> list[tuple[int, str]]:
    """같은 분기 10-Q/10-K 제출일 = d0인 신호의 (cik, accn) — 그 시각만 색인 페이지로 확인한다."""
    out = []
    for _, r in sig.iterrows():
        if pd.Timestamp(r["q_filing_date"]) == pd.Timestamp(r["d0"]):
            a = fd.q_accn_of(events, int(r["cik"]), r["report_date"], r["q_filing_date"])
            if a:
                out.append((int(r["cik"]), a))
    return out


def na_counts(F: pd.DataFrame) -> dict:
    return {f: {"판정 불가": int(F[f].isna().sum()), "사유": F.loc[F[f].isna(), f"{f}_why"].value_counts().to_dict()}
            for f in pf.FILTERS if f in F}


def step_fetch() -> None:
    """고가·저가·companyfacts·같은 분기 보고서 Accepted를 받고 판정 불가 개수만 출력한다(수익 계산 안 함)."""
    from data import edgar

    ua = edgar.get_user_agent()
    sig, qqq, days = main_signals()
    ev = pd.read_csv(dc.CACHE / "events.csv", dtype={"report_date": str})
    failed = fd.fetch_q_accepted(q_todo(sig, ev), ua)
    print(f"[fetch] 2012~2021 같은 분기 보고서 Accepted 실패 {len(failed)}: {failed[:20]}", flush=True)
    raw = {t: sx.load_ohlc(t) for t in sx.universe_tickers()}
    F, fails = main_flags(sig, qqq, raw)
    print(f"[fetch] 2012~2021 신호 {len(sig)} · 받기 실패 {len(fails)}: {fails[:30]}", flush=True)
    print(json.dumps(na_counts(F), ensure_ascii=False, indent=1), flush=True)
    s2, W2, prices, qq, pc = sealed_setup()
    ev2 = pd.read_csv(sd.CACHE / "events.csv", dtype={"report_date": str})
    failed = fd.fetch_q_accepted(q_todo(s2, ev2), ua)
    print(f"[fetch] 2022~ 같은 분기 보고서 Accepted 실패 {len(failed)}: {failed[:20]}", flush=True)
    F2, fails2 = sealed_flags(s2, prices, qq, pc, pf.FILTERS)
    print(f"[fetch] 2022~ 신호 {len(s2)} · 받기 실패 {len(fails2)}: {fails2[:30]}", flush=True)
    print(json.dumps(na_counts(F2), ensure_ascii=False, indent=1), flush=True)


# ── 계산 단계 ────────────────────────────────────────────────────────────────


def step_run() -> None:
    plan_commit = verify_plan()
    print(f"[pead-filters] 계획 태그 확인 ({plan_commit})", flush=True)
    sig, qqq, days = main_signals()
    print("[pead-filters] 신호 재현 636건 확인", flush=True)
    raw = {t: sx.load_ohlc(t) for t in sx.universe_tickers()}
    W = main_world(raw, qqq, days)
    F, fails = main_flags(sig, qqq, raw)
    sig = sig.join(F)

    res: dict = {"base": {}, "single": {}, "fail": {}, "na": {}, "judgeable": {}}
    for h in HOLDS:
        res["base"][h] = W.stats(sig, h, True)
        if (res["base"][h]["n"], round(res["base"][h]["mean"], 2)) != BASE_KNOWN[h]:
            raise SystemExit(f"[pead-filters] 기준 B1·{h} 재현 불일치 ({res['base'][h]['n']}건) — 멈춤")
    print("[pead-filters] 기준 B1·90 549건 +0.67% · B1·60 601건 +0.53% 재현 확인", flush=True)
    for f in pf.FILTERS:
        for d in ("single", "fail", "na", "judgeable"):
            res[d].setdefault(f, {})
        for h in HOLDS:
            res["single"][f][h] = W.stats(subset(sig, F, (f,), True), h, True)
            res["fail"][f][h] = W.stats(subset(sig, F, (f,), False), h, False)
            res["na"][f][h] = W.stats(subset(sig, F, (f,), None), h, False)
            if f in ("F4", "F5"):
                res["judgeable"][f][h] = W.stats(sig[F[f].notna()], h, False)
        print(f"[pead-filters] {f} 계산 끝", flush=True)

    base = res["base"][90]
    singles = [{"name": f, **res["single"][f][90]} for f in pf.FILTERS]
    passed, pair = pf.pick_pair(singles, base)
    combo = None
    if pair is not None:
        res["combo"] = {h: W.stats(subset(sig, F, pair, True), h, True) for h in HOLDS}
        combo = {"name": "+".join(pair), **res["combo"][90]}
        print(f"[pead-filters] 겹친 판 {combo['name']} 계산 끝 (1회)", flush=True)
    chosen, combo_ok = pf.final_choice(passed, combo, base)
    print(f"[pead-filters] 선택: {[c['name'] for c in chosen] or '필터 없음 유지'}", flush=True)

    s2, W2, prices, qq, pc = sealed_setup()
    ref = {"base": W2.stats(s2, 90, True)}
    if (ref["base"]["n"], round(ref["base"]["mean"], 2)) != SEALED_BASE_KNOWN:
        raise SystemExit(f"[pead-filters] 2022~ 기준 B1·90 재현 불일치 ({ref['base']['n']}건) — 멈춤")
    T2, _ = W2.trades(s2, 90)
    if not (T2["d0"].min() >= sd.D0_FIRST and T2["sell_day"].max() <= sa.SEAL_LAST):
        raise SystemExit("[pead-filters] 봉인 ⓐ 기간 밖 매매 — 멈춤")
    need = tuple(sorted({x for c in chosen for x in c["name"].split("+")}))
    sealed_na = {}
    if need:
        F2, fails2 = sealed_flags(s2, prices, qq, pc, need)
        fails += [("2022~ " + a, b) for a, b in fails2]
        for c in chosen:
            nm = tuple(c["name"].split("+"))
            ref[c["name"]] = W2.stats(subset(s2, F2, nm, True), 90, True)
            sealed_na[c["name"]] = int(F2[list(nm)].isna().any(axis=1).sum())
    print("[pead-filters] 2022~ 기준 B1·90 336건 −0.69% 재현 확인", flush=True)

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps({"res": res, "chosen": chosen, "ref": ref, "fails": fails}, ensure_ascii=False,
                                   indent=1, default=str), encoding="utf-8")
    write_report(res, singles, passed, pair, combo, combo_ok, chosen, ref, sealed_na, F, sig, fails, len(s2), plan_commit)
    print(f"[pead-filters] 완료 → {RESULT_MD.relative_to(ROOT)}", flush=True)


# ── 보고서 ───────────────────────────────────────────────────────────────────


def p(x, d: int = 2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{d}f}%"


def write_report(res, singles, passed, pair, combo, combo_ok, chosen, ref, sealed_na, F, sig, fails, n_sealed,
                 plan_commit) -> None:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    base = res["base"]
    rule = {r["name"]: pf.passes(r, base[90], 0.40) for r in singles}

    def row(label, S, h, show_pct=True):
        share = S["n"] / base[h]["n"] * 100 if base[h]["n"] else np.nan
        pc = f"{S['pct']:.1f}" if show_pct and np.isfinite(S.get("pct", np.nan)) else "—"
        return (f"| {label} | {S['n']} | {share:.0f}% | {p(S['mean'])} | {p(S['median'])} | "
                f"{S['win']:.0f}% | {pc} | {p(S['p1'])} / {p(S['p2'])} | {p(S['ex_pre'])} | "
                f"{p(S['ndx'])} ({S['n_ndx']}) | {p(S['sp'])} ({S['n_sp']}) |" if S["n"] else
                f"| {label} | 0 | 0% | — | — | — | — | — | — | — | — |")

    hdr = ["| 판 | 매매 | 기준 대비 | 평균 초과 | 중앙값 | 승률 | 무작위 백분위 | 2012~16 / 2017~21 | 비용 전(달러) | 나스닥100만 (건) | S&P100만 (건) |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    if chosen:
        sel = " · ".join(f"**{c['name']}**" for c in chosen)
    else:
        sel = "**필터 없음 유지**"
    L = ["# 실적급등 보완 필터 탐색 — 결과", "",
         f"> 계획: `{PLAN}` (태그 `{PLAN_TAG}` = 커밋 `{plan_commit}`, 실행 시 작업 폴더와 같음 확인) · 실행 코드 커밋 `{head}` · 계산 1회",
         "> 탐색이지 판정이 아니다. 원래 장부(pead-watch-prereg)는 바꾸지 않았다. 6개 필터를 한 번에 본 결과라 고른 판의 숫자는 위로 치우쳐 있다.", "",
         "- 신호: SW1 그대로, 반응일 2012~2021, 636건 재현 확인. 매매: B1(다음 날 시가) · 90거래일(주) / 60거래일(참고)",
         "- 초과 = 신호 종목 − 같은 날 같은 가격 종류로 사고판 QQQ, 원화 비용(수수료 0.07%·환전 0.1% 편도)·세금(매도 연도 장부 순이익 22%) 뒤",
         f"- 무작위: 같은 매수일·같은 예정 매도일, 매수일 전 달 말 명단 무작위 1종목, {ITERS}회, 시드 {SEED}",
         "- 점검: 기준 B1·90 549건 +0.67% · B1·60 601건 +0.53% · 2022~ B1·90 336건 −0.69% 재현 확인",
         "- 필터는 신호에 먼저 적용(걸러진 신호는 보유 중 겹침을 만들지 않음), 판정 불가는 통과로 치지 않음", "",
         f"## 선택: {sel}", ""]
    L += ["| 필터 | 2012~16 > 기준 ({}) | 2017~21 > 기준 ({}) | 매매 ≥ 기준 40% ({}건) | 통과 | 평균 초과 |".format(
        p(base[90]["p1"]), p(base[90]["p2"]), int(np.ceil(base[90]["n"] * 0.4))), "| --- | --- | --- | --- | --- | --- |"]
    for r in singles:
        q = rule[r["name"]]
        mk = lambda b: "✔" if b else "✗"  # noqa: E731
        L.append(f"| {NAMES[r['name']]} | {mk(q['p1_up'])} {p(r['p1'])} | {mk(q['p2_up'])} {p(r['p2'])} | "
                 f"{mk(q['enough'])} {r['n']} | {'**통과**' if q['ok'] else '—'} | {p(r['mean'])} |")
    L.append("")
    if not passed:
        L.append("- 통과한 단일 필터가 없다 → 계획 5절 4번대로 **필터 없음 유지**. 겹친 판은 계산하지 않았다.")
    elif pair is None:
        L.append(f"- 통과 1개({passed[0]['name']}) → 겹친 판은 계산하지 않았다(통과 2개 이상일 때만).")
    else:
        q = pf.passes(combo, base[90], 0.25)
        L.append(f"- 통과 {len(passed)}개 중 평균 초과 상위 2개 {pair[0]}·{pair[1]} 겹친 판(둘 다 통과): 매매 {combo['n']}"
                 f" ({combo['n'] / base[90]['n']:.0%}), 평균 {p(combo['mean'])}, 2012~16 {p(combo['p1'])} / 2017~21 {p(combo['p2'])}"
                 f" → {'후보' if combo_ok else '후보 아님'} (두 구간 {'✔' if q['p1_up'] and q['p2_up'] else '✗'}, 25% {'✔' if q['enough'] else '✗'})")
    if chosen:
        L.append("- 최종: 후보 중 평균 초과 큰 순 최대 2개 → " + ", ".join(f"{c['name']} {p(c['mean'])}" for c in chosen))
    for h in HOLDS:
        L += ["", f"## {h}거래일 보유 — 필터 통과 쪽", ""] + hdr
        L.append(row(f"기준(필터 없음) B1·{h}", base[h], h))
        for f in pf.FILTERS:
            L.append(row(NAMES[f], res["single"][f][h], h))
        if "combo" in res:
            L.append(row(f"겹친 판 {'+'.join(pair)}", res["combo"][h], h))
    for h in HOLDS:
        L += ["", f"## {h}거래일 보유 — 걸러진 쪽 · 판정 불가 쪽 (참고, 무작위 없음)", "",
              "| 필터 | 걸러진 쪽 매매 | 평균 초과 | 2012~16 / 2017~21 | 판정 불가 쪽 매매 | 평균 초과 |", "| --- | --- | --- | --- | --- | --- |"]
        for f in pf.FILTERS:
            a, b = res["fail"][f][h], res["na"][f][h]
            L.append(f"| {NAMES[f]} | {a['n']} | {p(a['mean'])} | {p(a['p1'])} / {p(a['p2'])} | {b['n']} | {p(b['mean'])} |")
    L += ["", "## 판정 불가 비율 (신호 636건 기준)", "", "| 필터 | 통과 | 걸러짐 | 판정 불가 | 사유 |", "| --- | --- | --- | --- | --- |"]
    for f in pf.FILTERS:
        v = F[f]
        na = v.isna()
        why = F.loc[na, f"{f}_why"].value_counts().to_dict()
        L.append(f"| {NAMES[f]} | {int((v == True).sum())} | {int((v == False).sum())} | {int(na.sum())} ({na.mean():.1%}) | "  # noqa: E712
                 + (", ".join(f"{a} {b}" for a, b in why.items()) or "—") + " |")
    ql = sig["q_late"].astype(bool)
    L += ["", f"- 같은 분기 10-Q/10-K가 반응일 장 마감(16:00 동부) 전에 나오지 않은 신호: {int(ql.sum())}/{len(sig)} ({ql.mean():.1%})."
          " 이 신호들은 보도자료 수치가 XBRL에 없어 F4·F5 판정 불가(사유 「분기 보고서 장 마감 뒤」)."]
    L += ["", "### F4·F5 판정 가능 신호만의 필터 없음 판 (참고)", "", "| 판 | 보유 | 매매 | 평균 초과 | 2012~16 / 2017~21 |", "| --- | --- | --- | --- | --- |"]
    for f in ("F4", "F5"):
        for h in HOLDS:
            S = res["judgeable"][f][h]
            L.append(f"| {NAMES[f]} 판정 가능 전체 | {h}일 | {S['n']} | {p(S['mean'])} | {p(S['p1'])} / {p(S['p2'])} |")
    L += ["", "## 2022 ~ 2026-09 (봉인 ⓐ, 이미 개봉) — 참고만, 선택에 안 씀", "",
          "| 판 | 매매 | 평균 초과 | 중앙값 | 승률 | 무작위 백분위 | 2022~23 / 2024~26 | 판정 불가 신호 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, S in ref.items():
        nm = "기준(필터 없음) B1·90" if k == "base" else k
        if S["n"] == 0:
            L.append(f"| {nm} | 0 | — | — | — | — | — | {sealed_na.get(k, '—')} |")
            continue
        L.append(f"| {nm} | {S['n']} | {p(S['mean'])} | {p(S['median'])} | {S['win']:.1f}% | {S['pct']:.1f} | "
                 f"{p(S['p1'])} / {p(S['p2'])} | {sealed_na.get(k, '—') if k != 'base' else '—'} |")
    L.append(f"- 봉인 ⓐ 신호 {n_sealed}건. 계획 6절대로 기준과 최종 선택 말고는 계산하지 않았다.")
    L += ["", "## 데이터 공백", "",
          f"- 받기 실패 {len(fails)}건" + (": " + ", ".join(f"{a}({b})" for a, b in fails[:40]) if fails else ""),
          "- 8-K 첨부 보도자료 수치는 무료 기계 판독(XBRL) 자료가 없어 쓰지 않았다 — F4·F5 판정 불가의 대부분이 이 때문.",
          "- 유료 데이터는 쓰지 않았다(Tiingo 무료 키, yfinance, EDGAR).",
          "", "## 결과를 본 뒤 바꾼 점", "", "없음", ""]
    RESULT_MD.write_text("\n".join(L), encoding="utf-8", newline="\n")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fetch":
        step_fetch()
    elif cmd == "run":
        step_run()
    else:
        raise SystemExit("사용: python -m scripts.pead_filters fetch|run")


if __name__ == "__main__":
    main()
