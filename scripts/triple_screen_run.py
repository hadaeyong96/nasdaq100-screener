"""3단 거르기 탐색 실행 (docs/prereg/triple_screen.md) — 탐색 구역, 판정 아님.

    python scripts/triple_screen_run.py --step counts   허용 조회만: 국면별 일수·회복장 구간 수·신호 건수 (수익 계산 없음)
    python scripts/triple_screen_run.py --step run      태그 triple-screen-prereg 확인 뒤 4판 + 비교 + 참고 → result.json
    python scripts/triple_screen_run.py --step run --quick   무작위 20회만 (실행기 점검, 보고에 쓰지 않음)

가격은 triple_screen_data.read_guarded(2021-12-31 넘는 줄에서 멈춤)로만 읽는다. 신호는 그날 대상(전 달 말 나스닥100)인 회사만.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import triple_screen_data as td  # noqa: E402
from core import tax_harvest as th  # noqa: E402
from core import triple_screen as ts  # noqa: E402

TAG = "triple-screen-prereg"
LOCKED = ["docs/prereg/triple_screen.md", "core/triple_screen.py", "scripts/triple_screen_data.py", "scripts/triple_screen_run.py"]
START, END = pd.Timestamp("2012-01-03"), pd.Timestamp("2021-12-31")
IND_FROM = pd.Timestamp("2011-01-01")
SEED = 20261019
N_RANDOM = 1000
FX_CSV = ROOT / "data" / "cache" / "shock_dip" / "fred_DEXKOUS.csv"
OUT = td.CACHE / "result.json"
MIDS = (100, 80, 120)


def verify_prereg() -> str:
    r = subprocess.run(["git", "rev-parse", "--verify", f"{TAG}^{{commit}}"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"태그 {TAG}가 없습니다 — 사전 등록 전에는 수익을 계산하지 않습니다")
    for f in LOCKED:
        if subprocess.run(["git", "diff", "--quiet", TAG, "--", f], cwd=ROOT).returncode != 0:
            raise SystemExit(f"{f}가 태그 {TAG} 뒤에 바뀌었습니다 — 멈춥니다")
    return r.stdout.strip()


def load_fx(days: pd.DatetimeIndex) -> np.ndarray:
    rows = []
    with open(FX_CSV, encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            d, v = line.strip().split(",")[:2]
            t = pd.Timestamp(d)
            if t > END:
                break
            if t >= pd.Timestamp("2010-06-01") and v not in ("", "."):
                rows.append((t, float(v)))
    s = pd.Series(dict(rows)).sort_index()
    out = s.reindex(s.index.union(days)).ffill().reindex(days)
    if out.isna().any():
        raise RuntimeError("환율 빈 날을 채우지 못했습니다")
    return out.to_numpy()


def etf(sym: str) -> pd.DataFrame:
    return td.read_guarded(td.PX / f"{sym}.csv")


def regimes_full(mids=MIDS) -> dict:
    """QQQ 1999-03 ~ 2021-12 전체로 국면(그날 종가까지). 키 (mid, use_ratio)."""
    q, xly, xlp = etf("QQQ")["close"], etf("XLY")["close"], etf("XLP")["close"]
    out = {}
    for m in mids:
        for ur in (True, False):
            out[(m, ur)] = ts.classify(q, xly.reindex(q.index).ffill(), xlp.reindex(q.index).ffill(), mid=m, use_ratio=ur)
    return out


def load_market() -> ts.Market:
    """지표 준비 2011-01 ~ 2021-12 시장 → 지표 계산 → 2012-01-03부터 자른 Market."""
    qd = etf("QQQ")
    days_full = qd.index[(qd.index >= IND_FROM) & (qd.index <= END)]
    comp = td.companies()
    rep = comp[comp["rep"]].sort_values("ticker")
    names = list(rep["ticker"])
    C = len(names)
    arr = {k: np.full((len(days_full), C), np.nan) for k in ("open", "high", "low", "close", "volume", "div")}
    for j, r in enumerate(rep.itertuples()):
        df = td.read_guarded(td.PX / f"{r.symbol}.csv", first=str(IND_FROM.date()))
        if r.ticker in td.LISTING:
            df = df[df.index >= td.LISTING[r.ticker]]
        df = df.reindex(days_full)
        for k in arr:
            arr[k][:, j] = df[k].to_numpy()
    qf = qd.reindex(days_full)  # 마지막 열 = 빈자리 돈 QQQ (대상 아님)
    for k in arr:
        arr[k] = np.column_stack([arr[k], qf[k].to_numpy()])
    names.append(ts.IDLE)
    arr["div"] = np.nan_to_num(arr["div"])
    # 대상: 그날이 속한 달의 전 달 말 명단에 그 회사(CIK) 주식이 하나라도 있으면
    tab = pd.read_csv(td.TABLE, encoding="utf-8-sig")
    cik_of = {t: int(c) for t, c in zip(tab["ticker"], tab["cik"]) if pd.notna(c)}
    col_of_cik = {int(c): j for j, c in enumerate(rep["cik"])}
    mem = td.members()
    member = np.zeros((len(days_full), C + 1), bool)
    pm = days_full.to_period("M")
    for me, tick in mem.items():
        use = (me + pd.offsets.MonthBegin(1)).to_period("M")
        rows = np.flatnonzero(pm == use)
        cols = {col_of_cik[cik_of[t]] for t in tick if t in cik_of and cik_of[t] in col_of_cik}
        for j in cols:
            member[rows, j] = True
    bil = etf("BIL").reindex(days_full)
    bret = ((bil["close"] + ts.KEEP_DIV * bil["div"].fillna(0.0)) / bil["close"].shift(1) - 1.0).fillna(0.0).to_numpy()
    qq = qd.reindex(days_full)
    mk = ts.Market(days_full, names, arr["open"], arr["high"], arr["low"], arr["close"], arr["volume"], arr["div"], member,
                   qq[["open", "close", "div"]].copy(), bret, load_fx(days_full))
    ts.build_indicators(mk, MIDS)
    s = int(np.searchsorted(days_full, START))
    cut = lambda a: a[s:]  # noqa: E731
    days = days_full[s:]
    m2 = ts.Market(days, names, cut(mk.open), cut(mk.high), cut(mk.low), cut(mk.close), cut(mk.volume), cut(mk.div),
                   cut(mk.member), qq.iloc[s:][["open", "close", "div"]].copy(), cut(bret), cut(mk.fx))
    m2.ind = {k: (v[s:] if isinstance(v, np.ndarray) else v) for k, v in mk.ind.items()}
    full = regimes_full()
    m2.reg = {k: v.reindex(days).to_numpy() for k, v in full.items()}
    m2.reg_full = full  # type: ignore[attr-defined]
    return m2


def cal_of(mk: ts.Market) -> ts.Calendar:
    return ts.calendar(mk.days, th.deadline)


# ── 허용 조회 ───────────────────────────────────────────────────────────


def step_counts() -> None:
    mk = load_market()
    reg = pd.Series(mk.reg[(100, True)], index=mk.days)
    print(f"[counts] 거래일 {len(mk.days)} ({mk.days[0].date()} ~ {mk.days[-1].date()}) · 회사 {len(mk.names) - 1}(+QQQ 빈자리 열)")
    print(f"[counts] 하루 평균 대상 회사 {mk.member.sum(axis=1).mean():.1f} · 대상 중 그날 종가 있음 "
          f"{(mk.member & ~np.isnan(mk.close)).sum() / mk.member.sum() * 100:.1f}%")
    for key in ((100, True), (100, False), (80, True), (120, True)):
        r = pd.Series(mk.reg[key], index=mk.days)
        vc = r.value_counts().to_dict()
        ep = ts.recovery_episodes(r)
        print(f"[counts] 국면 일수 mid={key[0]} XLY/XLP={'O' if key[1] else 'X'}: {vc} · 회복장 구간 {ep['episodes']}")
    ind = mk.ind
    valid = ~np.isnan(mk.close)
    with np.errstate(invalid="ignore"):
        st2 = mk.member & valid & ind["tide"] & (mk.close > ind["s200"])
        st2m = mk.member & valid & ind["tide"] & (mk.close > ind["s100"])
    up = reg.to_numpy() == ts.UP
    rec = reg.to_numpy() == ts.REC
    for trig in ("trig_rsi", "trig_stoch"):
        n_up = int((st2 & ind[trig])[up].sum())
        n_rec = int((st2m & ind[trig])[rec].sum())
        print(f"[counts] 신호 건수 {trig}: 상승장 {n_up} · 회복장(100일선 완화) {n_rec}")
    q = pd.Series(ts.classify(etf('QQQ')['close'], etf('XLY')['close'], etf('XLP')['close']))
    q = q[(q.index >= "1999-01-01") & (q.index <= "2011-12-31")]
    alt = alt_recovery()
    print(f"[counts] 회복장 대안 정의(50일선) 1999~2021: 일수 {alt['days']} · 구간 {alt['episodes']} · 연도별 일수 {alt['by_year']}")
    print(f"[counts] 참고 1999~2011 국면 일수: {q.value_counts().to_dict()} · 회복장 구간 {ts.recovery_episodes(q)['episodes']}")


def alt_recovery() -> dict:
    """별도 연구 후보 메모용(일수만, 수익 아님): QQQ < 200일선 & QQQ > 50일선 & 50일선 > 20거래일 전 50일선 & XLY/XLP > 그 비율 50일 평균.
    폭락장·상승장이 먼저(기본 국면과 같은 우선순위)."""
    q, xly, xlp = etf("QQQ")["close"], etf("XLY")["close"], etf("XLP")["close"]
    r = ts.classify(q, xly.reindex(q.index).ffill(), xlp.reindex(q.index).ffill(), mid=50, use_ratio=True)
    r = r[(r.index >= "1999-01-01") & (r.index <= "2021-12-31")]
    ep = ts.recovery_episodes(r)
    by = r[r == ts.REC].groupby(r[r == ts.REC].index.year).size().to_dict()
    return {"days": int((r == ts.REC).sum()), "episodes": ep["episodes"], "by_year": {int(k): int(v) for k, v in by.items()}}


# ── 실행 ────────────────────────────────────────────────────────────────

_POOL: dict = {}


def _init(mk, cal):
    _POOL.update(mk=mk, cal=cal)


def _rand(args) -> float:
    v, bpd, k = args
    r = ts.run_strategy(_POOL["mk"], v, ts.FULL, _POOL["cal"], rng=np.random.default_rng(SEED + k), buys_per_day=bpd)
    return ts.xirr(r["book"].flows)


def curves(res, mk) -> tuple[pd.Series, pd.Series]:
    b = res["book"]
    pre = pd.Series(b.curve_pre, index=mk.days)
    return pre, pd.Series(b.curve_stock, index=mk.days)


def yearly(c: pd.Series) -> dict[int, float]:
    out, prev = {}, float(c.iloc[0])
    for y in sorted(set(c.index.year)):
        last = float(c[c.index.year == y].iloc[-1])
        out[int(y)] = (last / prev - 1) * 100.0
        prev = last
    return out


def summarize(res, mk, reg: pd.Series) -> dict:
    b = res["book"]
    pre, stock = curves(res, mk)
    tr = pd.DataFrame(b.trades)
    out = {"xirr": ts.xirr(b.flows), "final_krw": res["final"], "mdd_pre": ts.mdd(pre),
           "taxes": {int(k): v for k, v in b.taxes.items()}, "years": yearly(pre),
           "regimes": ts.regime_table(pre, reg), "down_crash_mdd": ts.down_crash_mdd(pre, reg)}
    if len(tr) and "fill" in tr:
        tr = tr[tr["cost_krw"] > 0] if "cost_krw" in tr else tr
        hold = tr["exit"] - tr["fill"]
        out["trades"] = {"n": int(len(tr)), "avg_hold_days": float(hold.mean()), "win_pct": float((tr["pnl_krw"] > 0).mean() * 100),
                         "why": tr["why"].value_counts().to_dict(), "rec_bought": int(tr.get("rec", pd.Series(dtype=bool)).sum())}
        top = tr.sort_values("pnl_krw", ascending=False).head(5)
        fl = list(b.flows)
        fl[-1] = (fl[-1][0], fl[-1][1] - float(top["pnl_krw"].sum()))
        out["xirr_wo_top5"] = ts.xirr(fl)
        out["top5"] = [(mk.names[int(r.col)], str(mk.days[int(r.fill)].date()), str(mk.days[int(r.exit)].date()), float(r.pnl_krw))
                       for r in top.itertuples()]
    hv = stock
    out["turnover_pct"] = {int(y): b.sells_usd.get(y, 0.0) / float(hv[hv.index.year == y].mean()) * 100.0
                           for y in sorted(set(mk.days.year)) if float(hv[hv.index.year == y].mean()) > 0}
    if "stats" in res:
        out["stats"] = res["stats"]
    return out


def stages(fn) -> dict:
    return {name: ts.xirr(fn(c)["book"].flows) for name, c in (("세전", ts.ZERO), ("비용 뒤", ts.NO_TAX), ("세후", ts.FULL))}


def step_run(quick: bool) -> None:
    tag = verify_prereg()
    t0 = time.time()
    mk = load_market()
    cal = cal_of(mk)
    base_reg = pd.Series(mk.reg[(100, True)], index=mk.days)
    print(f"태그 {TAG} {tag[:7]} 확인 · 거래일 {len(mk.days)} · 회사 {len(mk.names) - 1} · 준비 {time.time() - t0:.0f}초", flush=True)
    out: dict = {"tag_commit": tag, "quick": quick}
    # 비교
    W_A = {ts.UP: 1.0}
    W_B = {ts.UP: 1.0, ts.REC: 0.30}
    rA, rB = mk.reg[(100, True)], mk.reg[(100, True)]
    comp = {
        "QQQ": lambda c: ts.run_qqq_hold(mk, c, cal),
        "1단만(A)": lambda c: ts.run_timing(mk, W_A, rA, c, cal),  # A판의 기준선: "세후 > 1단만" = 종목 고르기의 보탬
        "1단만(B)": lambda c: ts.run_timing(mk, W_B, rB, c, cal),
        "동일 비중": lambda c: ts.run_equal_weight(mk, c, cal),
    }
    out["compare"] = {}
    for k, fn in comp.items():
        res = fn(ts.FULL)
        s = summarize(res, mk, base_reg)
        s["stages"] = stages(fn)
        out["compare"][k] = s
        print(f"  {k}: 세후 {s['xirr']:.2f}%", flush=True)
    # 4판
    variants = [ts.A1, ts.A2]  # B·B0는 이번 시험에서 제외(태그 전 수정 2026-10-21 07:19)
    n_rand = 20 if quick else N_RANDOM
    out["variants"] = {}
    with ProcessPoolExecutor(initializer=_init, initargs=(mk, cal)) as ex:
        for v in variants:
            res = ts.run_strategy(mk, v, ts.FULL, cal)
            s = summarize(res, mk, base_reg)
            s["stages"] = stages(lambda c, v=v: ts.run_strategy(mk, v, c, cal))
            bpd = res["buys_per_day"]
            rand = list(ex.map(_rand, [(v, bpd, k) for k in range(n_rand)], chunksize=10))
            r = np.array(rand)
            x = s["xirr"]
            s["random"] = {"pct": float(100.0 * ((r < x).sum() + 0.5 * (r == x).sum()) / len(r)), "p50": float(np.median(r)),
                           "p95": float(np.percentile(r, 95)), "n": len(r)}
            out["variants"][v.name] = s
            print(f"  {v.name}: 세후 {x:.2f}% · 무작위 {s['random']['pct']:.1f}백분위 · {time.time() - t0:.0f}초", flush=True)
    # 회복장 신호
    out["recovery"] = {f"mid{k[0]}_{'ratio' if k[1] else 'noratio'}": ts.recovery_episodes(pd.Series(v, index=mk.days))
                       for k, v in mk.reg.items()}
    # 기준
    q = out["compare"]["QQQ"]
    crit = {}
    for name, s in out["variants"].items():
        tim = out["compare"]["1단만(A)"]
        c = {"세후 > QQQ": s["xirr"] > q["xirr"], "세후 > 1단만": s["xirr"] > tim["xirr"], "무작위 95백분위 이상": s["random"]["pct"] >= 95.0,
             "하락·폭락장 최대 낙폭이 QQQ보다 얕음": s["down_crash_mdd"] > q["down_crash_mdd"]}
        c["검증 제안 후보"] = all(c.values())
        crit[name] = c
    out["criteria"] = crit
    rec_days = int((base_reg == ts.REC).sum())
    out["rec_days_2012_2021"] = rec_days
    out["seconds"] = time.time() - t0
    if not quick:
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"끝 · {out['seconds']:.0f}초 · {OUT if not quick else '(quick, 저장 안 함)'}")


def main(argv=None) -> None:
    for st in (sys.stdout, sys.stderr):
        st.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["counts", "run"], required=True)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    step_counts() if a.step == "counts" else step_run(a.quick)


if __name__ == "__main__":
    main()
