"""탐색 B 1단계 실행 — B0~B3 계산·비교 (configs/sector_pullback_preregistration.yaml, 탐색 결과·판정 아님).

    python scripts/sector_pullback_run.py

시작할 때 잠근 파일(사전 등록·매핑표·데이터·core·이 스크립트)에 커밋 안 된 변경이 있으면 멈춘다(결과 보고 규칙 바꾸기 방지).
가격은 scripts/sector_pullback_data.read_guarded(2021-12-31 넘는 줄에서 멈춤)로만 읽는다.
결과: data/cache/sector_pullback/result.json(git 제외), docs/factory/explore/sector_pullback.md, docs/factory/trial_ledger.csv 4줄.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import sector_pullback as sp  # noqa: E402
from core.sector_map import SECTORS, UNMAPPED  # noqa: E402
from scripts import sector_pullback as smap  # noqa: E402
from scripts import sector_pullback_data as dat  # noqa: E402

LOCKED = ["configs/sector_pullback_preregistration.yaml", "configs/sector_pullback_sic_map.csv",
          "configs/sector_pullback_cik_candidates.csv", "configs/sector_pullback_sic_overrides.csv",
          "configs/sector_pullback_listing_dates.csv", "core/sector_pullback.py", "core/sector_map.py",
          "scripts/sector_pullback.py", "scripts/sector_pullback_data.py", "scripts/sector_pullback_run.py"]
START, END = pd.Timestamp("2012-01-03"), pd.Timestamp("2021-12-31")
MID = pd.Timestamp("2016-12-30")
INITIAL = 100_000.0
SEED = 20261016
N_RANDOM, N_FAKE = 1000, 200
RESULT = dat.CACHE / "result.json"
DOC = ROOT / "docs" / "factory" / "explore" / "sector_pullback.md"
LEDGER = ROOT / "docs" / "factory" / "trial_ledger.csv"
REGIMES = {0: "상승장", 1: "하락장", 2: "폭락장"}


def verify_locked() -> str:
    """잠근 파일에 커밋 안 된 변경이 없는지. 출력: HEAD 커밋. 아니면 SystemExit."""
    for f in LOCKED:
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", f], cwd=ROOT).returncode != 0:
            raise SystemExit(f"{f}에 커밋 안 된 변경이 있습니다 — 커밋한 규칙으로만 돌립니다")
        if subprocess.run(["git", "ls-files", "--error-unmatch", f], cwd=ROOT, capture_output=True).returncode != 0:
            raise SystemExit(f"{f}가 커밋돼 있지 않습니다")
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


# ── 시장 만들기 ──────────────────────────────────────────────────────────


def load_qqq() -> pd.DataFrame:
    q = dat.read_guarded(dat.yf_path("QQQ"))
    if q.index.max() != END:
        raise SystemExit(f"QQQ 마지막 날 {q.index.max().date()} ≠ 2021-12-31")
    return q


def companies_from(rows: pd.DataFrame) -> dict[int, dict]:
    """rows(ticker, cik, sector, path) → {cik: {tickers, rep, sector, df}} (회사당 한 종목, 사용자 결정 3)."""
    listing = pd.read_csv(dat.LISTING, comment="#")
    first = {r.ticker: pd.Timestamp(r.first_date) for r in listing.itertuples()}
    out = {}
    for cik, g in rows.groupby("cik"):
        shares = {}
        for r in g.itertuples():
            if isinstance(r.path, str) and r.path:
                df = dat.read_guarded(ROOT / r.path)
                if len(df):
                    shares[r.ticker] = df
        if not shares:
            continue
        rep, df = sp.combine_shares(shares, first)
        secs = set(g["sector"])
        if len(secs) != 1:
            raise SystemExit(f"CIK {cik}의 주식 업종이 다릅니다: {secs}")
        out[int(cik)] = {"tickers": sorted(g["ticker"]), "rep": rep, "sector": secs.pop(), "df": df}
    return out


def build_market(comps: dict[int, dict], qqq: pd.DataFrame, member_fn) -> sp.Market:
    """회사 표 → Market (QQQ 거래일 축). member_fn(days, ciks) → D×C bool."""
    days = qqq.index
    ciks = sorted(comps)
    close = np.full((len(days), len(ciks)), np.nan)
    div = np.zeros((len(days), len(ciks)))
    for j, c in enumerate(ciks):
        df = comps[c]["df"].reindex(days)
        close[:, j] = df["close"].to_numpy(dtype=float)
        div[:, j] = df["div"].fillna(0.0).to_numpy(dtype=float)
    sector = np.array([SECTORS.index(comps[c]["sector"]) if comps[c]["sector"] in SECTORS else -1 for c in ciks])
    mk = sp.Market(days=days, companies=[comps[c]["rep"] for c in ciks], close=close, div=div,
                   member=member_fn(days, ciks), sector=sector, sectors=list(SECTORS),
                   qclose=qqq["close"].to_numpy(dtype=float), qdiv=qqq["div"].to_numpy(dtype=float))
    sp.build_indicators(mk)
    return mk


def pit_market(qqq: pd.DataFrame) -> tuple[sp.Market, dict]:
    """시점별 명단 Market. 그날 대상 = 그달(전 달 말 명단)에 그 회사 주식이 하나라도 있으면."""
    src = pd.read_csv(dat.SOURCES)
    comps = companies_from(src.rename(columns={"sector": "sector"}))
    t2c = {r.ticker: int(r.cik) for r in src.itertuples()}
    members = smap.members_by_use_month()  # {그 달 말: 티커}

    def member_fn(days, ciks):
        col = {c: j for j, c in enumerate(ciks)}
        m = np.zeros((len(days), len(ciks)), dtype=bool)
        me = days.to_period("M").to_timestamp(how="end").normalize()
        for month, tickers in members.items():
            rows = np.flatnonzero(me == month)
            cols = sorted({col[t2c[t]] for t in tickers if t in t2c and t2c[t] in col})
            m[np.ix_(rows, cols)] = True
        return m

    mk = build_market(comps, qqq, member_fn)
    return mk, comps


def current_market(qqq: pd.DataFrame) -> tuple[sp.Market, dict]:
    """현재 명단 Market — 전 기간 같은 명단(가격이 있는 날만 실제로 대상이 된다)."""
    cur = pd.read_csv(dat.CURRENT)
    cur = cur[cur["cik"].notna()].assign(cik=lambda d: d["cik"].astype(int))
    comps = companies_from(cur)
    mk = build_market(comps, qqq, lambda days, ciks: np.ones((len(days), len(ciks)), dtype=bool))
    return mk, comps


# ── 성적 ─────────────────────────────────────────────────────────────────


def slice_cagr(curve: np.ndarray, days: pd.DatetimeIndex, d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    i0, i1 = days.get_loc(d0), days.get_loc(d1)
    return sp.cagr(curve[i0], curve[i1], days[i0], days[i1])


def yearly(curve: np.ndarray, days: pd.DatetimeIndex) -> dict[int, float]:
    s = pd.Series(curve, index=days)
    out, prev = {}, float(s.iloc[0])
    for y in range(days[0].year, days[-1].year + 1):
        v = float(s[s.index.year == y].iloc[-1])
        out[y] = (v / prev - 1.0) * 100.0
        prev = v
    return out


def regime_perf(curve: np.ndarray, reg: np.ndarray) -> dict[str, dict]:
    """그날 국면에 그날 → 다음 날 수익. 출력 {국면: {days, cum_pct, ann_pct}}."""
    r = curve[1:] / curve[:-1] - 1.0
    g = reg[:-1]
    out = {}
    for k, name in REGIMES.items():
        x = r[g == k]
        if len(x) == 0:
            out[name] = {"days": 0, "cum_pct": None, "ann_pct": None}
            continue
        cum = float(np.prod(1 + x))
        out[name] = {"days": int(len(x)), "cum_pct": (cum - 1) * 100, "ann_pct": (cum ** (252 / len(x)) - 1) * 100}
    return out


def summarize(name: str, r: dict, q: dict, mk: sp.Market, s: int, e: int, reg: np.ndarray) -> dict:
    days = mk.days[s:e + 1]
    out = {"name": name, "final": r["final"], "cagr": sp.cagr(INITIAL, r["final"], days[0], days[-1]),
           "mdd": sp.mdd(r["curve"]), "taxes": r["taxes"]}
    out["p1"] = slice_cagr(r["curve"], days, START, MID)
    out["p2"] = slice_cagr(r["curve"], days, MID, END)
    out["q_p1"] = slice_cagr(q["curve"], days, START, MID)
    out["q_p2"] = slice_cagr(q["curve"], days, MID, END)
    ys, yq = yearly(r["curve"], days), yearly(q["curve"], days)
    out["yearly"] = {y: [ys[y], yq[y]] for y in ys}
    out["years_beat"] = sum(1 for y in ys if ys[y] > yq[y])
    tr = r.get("trades")
    if tr is not None:
        out["n_trades"] = len(tr)
        out["avg_hold"] = float(np.mean([t["sell"] - t["buy"] for t in tr])) if tr else None
        out["win_rate"] = float(np.mean([t["pnl"] > 0 for t in tr]) * 100) if tr else None
        out["no_slot"], out["no_fill"] = len(r["no_slot"]), len(r["no_fill"])
        by_sec = defaultdict(float)
        for t in tr:
            k = mk.sector[t["c"]]
            by_sec[SECTORS[k] if k >= 0 else UNMAPPED] += t["pnl"]
        by_sec["QQQ(쉬는 돈)"] = r["qqq_pnl"]
        out["sector_pnl"] = dict(sorted(by_sec.items(), key=lambda kv: -kv[1]))
    out["regime"] = regime_perf(r["curve"], reg[s:e + 1])
    return out


def pct_rank(x: float, arr: list[float]) -> float:
    a = np.asarray(arr)
    return 100.0 * (float((a < x).sum()) + 0.5 * float((a == x).sum())) / len(a)


def run_universe(mk: sp.Market, label: str, full: bool) -> dict:
    """한 명단에서 B0~B3·QQQ·동일 비중 (full이면 무작위·가짜·0.3%·상위 5건 제외까지)."""
    s, e = mk.days.get_loc(START), mk.days.get_loc(END)
    reg = sp.regime(mk.qclose)
    c1, c3 = sp.Costs(0.001), sp.Costs(0.003)
    q1, ew1 = sp.run_qqq(mk, s, e, c1), sp.run_equal_weight(mk, s, e, c1)
    res = {"label": label, "qqq": summarize("QQQ", q1, q1, mk, s, e, reg), "ew": summarize("동일 비중", ew1, q1, mk, s, e, reg),
           "variants": {}}
    if full:
        q3, ew3 = sp.run_qqq(mk, s, e, c3), sp.run_equal_weight(mk, s, e, c3)
        res["qqq_03"] = sp.cagr(INITIAL, q3["final"], mk.days[s], mk.days[e])
        res["ew_03"] = sp.cagr(INITIAL, ew3["final"], mk.days[s], mk.days[e])
    vds, evs, exs = {}, {}, {}
    for v in sp.VARIANTS:
        t0 = time.time()
        vd = sp.variant_data(mk, v)
        ev = sp.signal_events(mk, vd, s, e)
        ex = sp.exit_flags(mk, vd)
        vds[v], evs[v], exs[v] = vd, ev, ex
        r1 = sp.run_book(mk, s, e, c1, events=ev, exits=ex)
        out = summarize(v, r1, q1, mk, s, e, reg)
        out["n_signals"] = len(ev)
        if full:
            r3 = sp.run_book(mk, s, e, c3, events=ev, exits=ex)
            out["cagr_03"] = sp.cagr(INITIAL, r3["final"], mk.days[s], mk.days[e])
            top5 = sorted(r1["trades"], key=lambda t: -t["pnl"])[:5]
            keep = [{"c": t["c"], "buy": t["buy"], "sell": t["sell"]} for t in r1["trades"] if t not in top5]
            rx = sp.run_book(mk, s, e, c1, schedule=keep)
            out["cagr_ex_top5"] = sp.cagr(INITIAL, rx["final"], mk.days[s], mk.days[e])
            out["top5"] = [{"company": mk.companies[t["c"]], "buy": str(mk.days[t["buy"]].date()),
                            "sell": str(mk.days[t["sell"]].date()), "pnl": t["pnl"]} for t in top5]
            rng = np.random.default_rng([SEED, sp.VARIANTS.index(v)])
            rc = []
            for _ in range(N_RANDOM):
                sch = sp.random_schedule(mk, r1["trades"], rng)
                rr = sp.run_book(mk, s, e, c1, schedule=sch)
                rc.append(sp.cagr(INITIAL, rr["final"], mk.days[s], mk.days[e]))
            out["random_pct"] = pct_rank(out["cagr"], rc)
            out["random_median"] = float(np.median(rc))
            out["random_p95"] = float(np.percentile(rc, 95))
        res["variants"][v] = out
        print(f"[{label}] {v}: 신호 {len(ev)} · 매매 {out['n_trades']} · 세후 연복리 {out['cagr']:.2f}% ({time.time() - t0:.0f}초)",
              flush=True)
    if full:
        rng = np.random.default_rng([SEED, 99])
        best = []
        for k in range(N_FAKE):
            vals = []
            for v in sp.VARIANTS:
                fe = sp.fake_events(mk, len(evs[v]), s, e, rng)
                rr = sp.run_book(mk, s, e, c1, events=fe, exits=exs[v])
                vals.append(sp.cagr(INITIAL, rr["final"], mk.days[s], mk.days[e]))
            best.append(max(vals))
            if (k + 1) % 50 == 0:
                print(f"[{label}] 가짜 {k + 1}/{N_FAKE}", flush=True)
        res["fake_top_p95"] = float(np.percentile(best, 95))
        res["fake_top_median"] = float(np.median(best))
    return res


def jumps(mk: sp.Market) -> list[dict]:
    """대상인 날 일간 ±50% 넘는 움직임 (분사 등, 고치지 않고 보고만)."""
    out = []
    for j in range(mk.close.shape[1]):
        vi = np.flatnonzero(~np.isnan(mk.close[:, j]))
        for a, b in zip(vi[:-1], vi[1:]):
            r = mk.close[b, j] / mk.close[a, j] - 1
            if abs(r) > 0.5 and mk.member[b, j] and mk.days[b] >= START:
                out.append({"company": mk.companies[j], "date": str(mk.days[b].date()), "ret_pct": r * 100})
    return out


def coverage(mk: sp.Market, comps: dict) -> dict:
    """대상(회사·날) 중 종가가 있는 비율, 회사 수."""
    s, e = mk.days.get_loc(START), mk.days.get_loc(END)
    m = mk.member[s:e + 1]
    ok = m & ~np.isnan(mk.close[s:e + 1])
    return {"companies": len(comps), "member_company_days": int(m.sum()), "with_price_pct": float(ok.sum() / m.sum() * 100),
            "avg_members_per_day": float(m.sum(axis=1).mean()), "avg_with_price_per_day": float(ok.sum(axis=1).mean())}


# ── 기록 ─────────────────────────────────────────────────────────────────


def f(x, d=2, sign=False):
    if x is None:
        return "-"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def write_doc(R: dict, head: str) -> list[dict]:
    P, C = R["pit"], R["current"]
    q, ew = P["qqq"], P["ew"]
    thr = P["fake_top_p95"]
    L = [f"# 탐색 B 「강한 업종 속 강한 종목, 첫 눌림 매수」 — 탐색 결과(판정 아님)", "",
         f"- 사전 등록 `configs/sector_pullback_preregistration.yaml` (실행 커밋 `{head}`, 브랜치 research/sector-pullback). "
         f"실행 `python scripts/sector_pullback_run.py` ({R['run_at']})",
         "- 대상: 시점별 나스닥100 ∪ S&P 100 (전 달 말 명단, 회사당 한 종목), 2012-01-03 ~ 2021-12-31, 2022년 이후 데이터 가드",
         f"- 확보: 회사 {P['coverage']['companies']}곳, 대상 회사·날 중 종가 있음 {P['coverage']['with_price_pct']:.1f}% "
         f"(하루 평균 대상 {P['coverage']['avg_members_per_day']:.1f} · 가격 있음 {P['coverage']['avg_with_price_per_day']:.1f})",
         "- 비용 편도 0.1%(종목·QQQ 모두), 세금 연 실현 순이익 22%(이월 없음), 분배금 원천 15% 뒤 현금 → 쉬는 돈 QQQ", "",
         "## 변형별 성적 (비용 0.1%, 세후)", "",
         "| | B0 기본형 | B1 돌파 다음 날 | B2 업종 조건 없음 | B3 업종 63일 | QQQ | 동일 비중 |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    V = [P["variants"][v] for v in sp.VARIANTS]

    def row(label, fn, qv="", ev=""):
        L.append(f"| {label} | " + " | ".join(fn(x) for x in V) + f" | {qv} | {ev} |")

    row("세후 연복리(%)", lambda x: f(x["cagr"]), f(q["cagr"]), f(ew["cagr"]))
    row("QQQ 대비(%p)", lambda x: f(x["cagr"] - q["cagr"], sign=True))
    row("동일 비중 대비(%p)", lambda x: f(x["cagr"] - ew["cagr"], sign=True), f(q["cagr"] - ew["cagr"], sign=True))
    row("무작위 1,000회 백분위", lambda x: f(x["random_pct"], 1))
    row("무작위 중앙값·95백분위(%)", lambda x: f"{f(x['random_median'])} · {f(x['random_p95'])}")
    row(f"가짜 1등 95백분위 {thr:.2f}% 대비(%p)", lambda x: f(x["cagr"] - thr, sign=True))
    row("2012~16 연복리(%)", lambda x: f(x["p1"]), f(q["p1"]), f(ew["p1"]))
    row("2017~21 연복리(%)", lambda x: f(x["p2"]), f(q["p2"]), f(ew["p2"]))
    row("연도별 QQQ 이긴 해", lambda x: f"{x['years_beat']}/10", "", f"{ew['years_beat']}/10")
    row("최대 낙폭(%)", lambda x: f(x["mdd"]), f(q["mdd"]), f(ew["mdd"]))
    row("신호 수", lambda x: str(x["n_signals"]))
    row("매매 수 · 자리 없음 · 체결 불가", lambda x: f"{x['n_trades']} · {x['no_slot']} · {x['no_fill']}")
    row("평균 보유일(거래일)", lambda x: f(x["avg_hold"], 1))
    row("승률(%)", lambda x: f(x["win_rate"], 1))
    row("상위 5건 제외 연복리(%)", lambda x: f(x["cagr_ex_top5"]))
    row("비용 0.3% 연복리(%)", lambda x: f(x["cagr_03"]), f(P["qqq_03"]), f(P["ew_03"]))
    L += ["", f"가짜 신호 조립: 회차마다 4개 변형을 가짜 신호(실제 신호 수만큼 무작위 날짜·회사, 같은 매도·자리 규칙)로 돌려 가장 높은 값 = 가짜 1등. "
          f"200회 중앙값 {P['fake_top_median']:.2f}%, 95백분위 **{thr:.2f}%**.", "",
          "## 연도별 수익률 (%, 전략 / QQQ)", "", "| 연도 | " + " | ".join(sp.VARIANTS) + " | 동일 비중 | QQQ |",
          "| --- | " + " | ".join(["---"] * 6) + " |"]
    for y in range(2012, 2022):
        L.append(f"| {y} | " + " | ".join(f(x["yearly"][y][0], 1) for x in V) + f" | {f(ew['yearly'][y][0], 1)} | {f(q['yearly'][y][0], 1)} |")
    L += ["", "(2012는 2012-01-03 종가부터. 전략 곡선은 연말 세금을 그해에 내고 QQQ·동일 비중 곡선은 끝에만 세금 — 구간·연도 비교는 전략에 불리)", "",
          "## 국면별 성적 (세전 일별 곡선, 그날 국면 → 다음 날 수익, 연환산 %)", "",
          "| 국면 | 일수 | " + " | ".join(sp.VARIANTS) + " | 동일 비중 | QQQ |", "| --- | --- | " + " | ".join(["---"] * 6) + " |"]
    for name in REGIMES.values():
        L.append(f"| {name} | {q['regime'][name]['days']} | " + " | ".join(f(x["regime"][name]["ann_pct"], 1) for x in V)
                 + f" | {f(ew['regime'][name]['ann_pct'], 1)} | {f(q['regime'][name]['ann_pct'], 1)} |")
    L += ["", "## 업종별 기여 (매매 손익 합, 비용 뒤·세전, $; 처음 $100,000)", ""]
    secs = sorted({k for x in V for k in x["sector_pnl"]}, key=lambda k: -sum(x["sector_pnl"].get(k, 0) for x in V))
    L += ["| 업종 | " + " | ".join(sp.VARIANTS) + " |", "| --- | " + " | ".join(["---"] * 4) + " |"]
    for k in secs:
        L.append(f"| {k} | " + " | ".join(f"{x['sector_pnl'].get(k, 0):,.0f}" for x in V) + " |")
    L += ["", "## 손익 상위 5건 (변형별)", ""]
    for v, x in zip(sp.VARIANTS, V):
        L.append(f"- {v}: " + "; ".join(f"{t['company']} {t['buy']}→{t['sell']} ${t['pnl']:,.0f}" for t in x["top5"]))
    L += ["", "## 같은 규칙 · 현재 명단 (생존 편향 있음, 참고)", "",
          f"- 명단: 나스닥100(2026-09-23) ∪ S&P 100(위키 현재 판) 전 기간 고정, 회사 {C['coverage']['companies']}곳",
          "", "| | " + " | ".join(sp.VARIANTS) + " | QQQ | 동일 비중 |", "| --- | " + " | ".join(["---"] * 6) + " |"]
    CV = [C["variants"][v] for v in sp.VARIANTS]
    L.append("| 세후 연복리(%) — 시점별 | " + " | ".join(f(x["cagr"]) for x in V) + f" | {f(q['cagr'])} | {f(ew['cagr'])} |")
    L.append("| 세후 연복리(%) — 현재 명단 | " + " | ".join(f(x["cagr"]) for x in CV) + f" | {f(C['qqq']['cagr'])} | {f(C['ew']['cagr'])} |")
    L.append("| 현재 명단 QQQ 대비(%p) | " + " | ".join(f(x["cagr"] - C["qqq"]["cagr"], sign=True) for x in CV) + " | | |")
    L.append("| 현재 명단 2012~16 / 2017~21 | " + " | ".join(f"{f(x['p1'])} / {f(x['p2'])}" for x in CV)
             + f" | {f(C['qqq']['p1'])} / {f(C['qqq']['p2'])} | {f(C['ew']['p1'])} / {f(C['ew']['p2'])} |")
    L.append("| 현재 명단 매매 수 · 최대 낙폭 | " + " | ".join(f"{x['n_trades']} · {f(x['mdd'], 1)}" for x in CV) + " | | |")
    # 다음 단계 기준 (미리 정함)
    L += ["", "## 다음 단계 기준 점검 (미리 정한 기준: 가짜 1등 95백분위 초과 + 동일 비중보다 높음 + 두 구간 모두 QQQ 초과)", "",
          "| 변형 | 가짜 1등 95백분위 초과 | 동일 비중보다 높음 | 2012~16 QQQ 초과 | 2017~21 QQQ 초과 | 셋 다 |", "| --- | --- | --- | --- | --- | --- |"]
    led = []
    any_pass = False
    for v, x in zip(sp.VARIANTS, V):
        a, b, c, d = x["cagr"] > thr, x["cagr"] > ew["cagr"], x["p1"] > q["p1"], x["p2"] > q["p2"]
        ok = a and b and c and d
        any_pass |= ok
        L.append(f"| {v} | {'O' if a else 'X'} | {'O' if b else 'X'} | {'O' if c else 'X'} | {'O' if d else 'X'} | **{'충족' if ok else '미충족'}** |")
        led.append({"date": date.today().isoformat(), "id": "sector_pullback", "variant": v, "zone": "탐색",
                    "universe": "시점별 나스닥100 ∪ S&P 100", "period": "2012-01~2021-12", "status": "탐색 결과(판정 아님)",
                    "cagr_after_tax": round(x["cagr"], 2), "qqq_cagr": round(q["cagr"], 2), "ew_cagr": round(ew["cagr"], 2),
                    "random_pct": round(x["random_pct"], 1), "fake_top_p95": round(thr, 2), "trades": x["n_trades"],
                    "next_step": "검증 구역 ③ 판정 제안" if ok else "기준 미충족", "commit": head})
    L += ["", ("**기준을 충족한 변형이 있다 → 검증 구역 ③ 판정 제안만 하고 멈춘다 (판정 아님).**" if any_pass else
               "**기준을 모두 충족한 변형이 없다 → 검증 구역 ③ 판정 제안 없음.**"), "",
          "## 한계", "",
          "- 생존 편향: 가격 없는 23종목(CA·EMC·YHOO·SNDK·MON·QRTEA·BBBY 등, 명단 종목·달의 약 4%)과 0단계 제외 4종목은 빠졌다",
          "- 업종은 현재 SIC 하나를 전 기간에 쓴다. 부동산은 거의 늘 3종목 미만이라 순위에서 빠진다",
          "- 분할 반영 종가만 쓴다(분사 미반영). 일간 ±50% 넘는 움직임: " + (", ".join(f"{j['company']} {j['date']} {j['ret_pct']:+.0f}%" for j in R["jumps"]) or "없음"),
          "- SW1 Tiingo 출처 15종목은 분배금 없음, Tiingo 출처는 2011-06부터라 지표 준비 기간이 짧다",
          "- 해석(가짜 신호 조립 방식, 첫 눌림에서 조건이 안 맞으면 그 돌파는 끝 등)은 사전 등록 step1_interpretations 그대로"]
    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")
    return led


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    head = verify_locked()
    print(f"[run] 잠근 파일 확인 통과 (HEAD {head})", flush=True)
    t0 = time.time()
    qqq = load_qqq()
    mk, comps = pit_market(qqq)
    cmk, ccomps = current_market(qqq)
    print(f"[run] 시점별 회사 {len(comps)} · 현재 명단 회사 {len(ccomps)} ({time.time() - t0:.0f}초)", flush=True)
    R = {"run_at": time.strftime("%Y-%m-%d %H:%M"), "head": head}
    R["pit"] = run_universe(mk, "시점별", full=True)
    R["pit"]["coverage"] = coverage(mk, comps)
    R["current"] = run_universe(cmk, "현재 명단", full=False)
    R["current"]["coverage"] = coverage(cmk, ccomps)
    R["jumps"] = jumps(mk)
    dat.CACHE.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    led = write_doc(R, head)
    old = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(led)], ignore_index=True).to_csv(LEDGER, index=False, encoding="utf-8-sig")
    print(f"[run] 끝 ({time.time() - t0:.0f}초) → {DOC.relative_to(ROOT)}, {LEDGER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
