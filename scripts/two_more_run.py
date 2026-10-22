"""전략 2개 더 — 「주식·채권·금 갈아타기」·「주식·채권 2배 균형」 성적 (docs/prereg/two_more.md, 태그 two-more-prereg).

모든 데이터는 2021-12-31에서 자른다(scripts/sealed_guard + 새 자료는 요청 단계부터 2021-12-31까지). 결과를 본 뒤 규칙을 바꾸지 않는다.
실행: python scripts/two_more_run.py   → docs/research/two_more_result.json (표는 scripts/three_strategies_doc.py가 문서로)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import sealed_guard  # noqa: E402

sealed_guard.install()

from core import multi_book as mb  # noqa: E402
from core import shock_dip as sd  # noqa: E402
from core import shock_dip_perf as sp  # noqa: E402
from core import two_more as tm  # noqa: E402
from scripts.shock_dip_gate01 import CACHE  # noqa: E402
from scripts.shock_dip_gate2 import load_close  # noqa: E402
from scripts.shock_dip_gate3 import START_KRW, asof_series, assets  # noqa: E402

OUT = ROOT / "docs" / "research"
COST = 0.001
N_RANDOM = int(os.environ.get("TM_N_RANDOM", "1000"))  # 점검용으로만 줄인다(판정은 1000)
SEED = 20261022
SUBS = {"S&P500": [("1977~87", "1977-01-01", "1987-12-31"), ("1988~98", "1988-01-01", "1998-12-31")],
        "QQQ(보정)": [("2000~10", "2000-01-01", "2010-12-31"), ("2011~21", "2011-01-01", "2021-12-31")]}


def _sealed_csv(name: str) -> pd.Series:
    s = pd.read_csv(CACHE / name, index_col=0, parse_dates=True).iloc[:, 0]
    return sealed_guard.cut(pd.to_numeric(s, errors="coerce"))


def _sealed_yf(tk: str, col: str = "Adj Close") -> pd.Series:
    df = pd.read_csv(CACHE / f"sealed_yf_{tk}.csv", index_col=0, parse_dates=True)
    return sealed_guard.cut(df[col].dropna())


def build(name: str, dtb3: pd.Series) -> dict:
    """구간 하나의 가격 행렬(열: 현금·주식1·장기채1·금·주식2·장기채2)과 금 시작일."""
    if name == "QQQ(보정)":
        close, _ = sd.correct_with_reference(load_close("QQQ"), load_close("^NDX"), 0.02)
        close = close[close.index <= "2021-12-31"]
        stock, stock2, _, _ = assets(name, "QQQ", close, "2021-12-31", dtb3)
        d = close.index
    else:
        tk = "^GSPC" if name == "S&P500" else "^IXIC"
        close = load_close(tk)
        close = close[(close.index <= "1998-12-31") & (close.index >= "1962-01-02")]  # 장기채 금리(DGS10) 시작 뒤
        d = close.index
        if name == "S&P500":
            sh = pd.read_csv(CACHE / "sealed_shiller_PD.csv", index_col=0, parse_dates=True)
            stock = tm.total_return_with_monthly_dividend(close, sealed_guard.cut(sh["D"]))
        else:
            stock = close / close.iloc[0]  # 나스닥 종합: 배당 자료 없음(참고만)
        stock2, _ = sp.splice_leveraged(stock.pct_change().fillna(0.0), None, asof_series(dtb3, d))
    rate = asof_series(dtb3, d)
    cash = (1 + rate.shift(1).fillna(0) / 100 / 252).cumprod()
    bond_r = tm.bond_returns(asof_series(_sealed_csv("sealed_fred_DGS10.csv"), d))
    bond = tm.splice_real(bond_r, _sealed_yf("TLT") if name == "QQQ(보정)" else None)
    bond2, _ = sp.splice_leveraged(bond.pct_change().fillna(0.0), None, rate)
    gold = pd.Series(1.0, index=d)
    gold_start = None
    if name == "QQQ(보정)":
        gc = _sealed_yf("GC_F", "Close")
        g = asof_series(gc, d)
        gold_start = gc.index[0]
        gr = g.pct_change().where(d > gold_start, 0.0).fillna(0.0)
        gold = tm.splice_real(gr, _sealed_yf("GLD"))
    prices = np.column_stack([cash, stock.reindex(d), bond, gold, stock2.reindex(d), bond2]).astype(float)
    return {"d": d, "prices": prices, "gold_start": gold_start}


def book(m, start, end, decisions, should, fx_all, start_w, tax):
    d = m["d"]
    w0 = int(np.searchsorted(d, pd.Timestamp(start)))
    w1 = int(np.searchsorted(d, pd.Timestamp(end), side="right"))
    px = m["prices"][w0:w1]
    dd = d[w0:w1]
    dec = {p - w0: w for p, w in decisions.items() if w0 <= p < w1}
    kw = dict(start_krw=1.0)
    if tax:
        kw = dict(tax=True, fx=asof_series(fx_all, dd).to_numpy(), start_krw=START_KRW, may_pay=sp.may_payment_positions(dd),
                  years=dd.year.to_numpy(), sell_order=[tm.CASH, tm.GOLD, tm.BOND, tm.STOCK, tm.BOND2, tm.STOCK2])
    r = mb.run_multi(px, list(start_w), dec, should, cost=COST, **kw)
    days = (dd[-1] - dd[0]).days
    pre = mb.cagr(r.nav[0], r.end_value, days)
    val = mb.cagr(START_KRW, r.end_after_tax_krw, days) if tax else pre
    return {"val": val, "pre": pre, "mdd": mb.max_drawdown(r.nav), "trades": r.trades, "nav": r.nav, "cash": px[:, tm.CASH],
            "window": f"{dd[0].date()} ~ {dd[-1].date()}"}


def changed(cur, tgt):
    return float(np.abs(cur - tgt).max()) > 1e-9


def switch_decisions(m, start, months, use_gold=True):
    d, prices = m["d"], m["prices"]
    w0 = int(np.searchsorted(d, pd.Timestamp(start)))
    first = {tm.GOLD: m["gold_start"]} if use_gold and m["gold_start"] is not None else {}
    assets_ = (tm.STOCK, tm.BOND, tm.GOLD, tm.CASH) if use_gold and m["gold_start"] is not None else (tm.STOCK, tm.BOND, tm.CASH)
    dec, cur = {}, tm.STOCK
    for p in tm.month_end_positions(d, w0):
        c = tm.switch_choice(tm.lookback_returns(prices, d, p, months, first, assets_))
        if c is not None:
            cur = c
        dec[p] = tm.one_hot(cur)
    return dec


def random_switch(m, start, end, k, rng, use_gold=True):
    d = m["d"]
    w0 = int(np.searchsorted(d, pd.Timestamp(start)))
    w1 = int(np.searchsorted(d, pd.Timestamp(end), side="right"))
    pos = [p for p in tm.month_end_positions(d, w0) if p < w1 - 1]
    chosen = set(rng.choice(len(pos), size=min(k, len(pos)), replace=False).tolist())
    dec, cur = {}, tm.STOCK
    for i, p in enumerate(pos):
        if i in chosen:
            cands = [tm.CASH, tm.STOCK, tm.BOND] + ([tm.GOLD] if use_gold and m["gold_start"] is not None and d[p] >= m["gold_start"] else [])
            cur = int(rng.choice([a for a in cands if a != cur]))
        dec[p] = tm.one_hot(cur)
    return dec


def stock_hold(m, start, end, fx_all, tax):
    return book(m, start, end, {}, lambda a, b: False, fx_all, tm.one_hot(tm.STOCK), tax)


def balanced_weights(a, lev=True):
    w = np.zeros(6)
    if lev:
        w[tm.STOCK2], w[tm.BOND2] = a, 1 - a
    else:
        w[tm.STOCK], w[tm.BOND] = a, 1 - a
    return w


def balanced(m, start, end, fx_all, tax, a=0.55, lev=True):
    d = m["d"]
    w0 = int(np.searchsorted(d, pd.Timestamp(start)))
    tgt = balanced_weights(a, lev)
    col = tm.STOCK2 if lev else tm.STOCK
    dec = {p: tgt for p in tm.quarter_end_positions(d, w0)}
    return book(m, start, end, dec, lambda cur, t: abs(cur[col] - t[col]) >= 0.05 - 1e-12, fx_all, tgt, tax)


def main() -> None:
    dtb3 = sealed_guard.load_fred("DTB3")
    fx_all = sealed_guard.load_fred("DEXKOUS")
    rng = np.random.default_rng(SEED)
    mk = {n: build(n, dtb3) for n in ("S&P500", "나스닥 종합", "QQQ(보정)")}
    win = {"S&P500": ("1977-01-01", "1998-12-31"), "나스닥 종합": ("1977-01-01", "1998-12-31"), "QQQ(보정)": ("2000-01-01", "2021-12-31")}
    out = {"switch": {}, "balanced": {}, "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                                                  text=True, cwd=ROOT).stdout.strip()}
    for n, m in mk.items():
        s, e = win[n]
        tax = n == "QQQ(보정)"
        base = stock_hold(m, s, e, fx_all, tax)
        rec = {"window": base["window"], "basis": "세후 원화" if tax else "세전", "stock_cagr": base["val"], "stock_mdd": base["mdd"],
               "judged": n == "QQQ(보정)", "gold": m["gold_start"] is not None}
        # 갈아타기 (S&P·나스닥은 금 없이 참고)
        r12 = book(m, s, e, switch_decisions(m, s, 12), changed, fx_all, tm.one_hot(tm.STOCK), tax)
        rec.update(cagr=r12["val"], mdd=r12["mdd"], switches=r12["trades"])
        for mo in (9, 6):
            rec[f"cagr_{mo}m"] = book(m, s, e, switch_decisions(m, s, mo), changed, fx_all, tm.one_hot(tm.STOCK), tax)["val"]
        if n == "QQQ(보정)":
            fakes = np.array([book(m, s, e, random_switch(m, s, e, r12["trades"], rng), changed, fx_all, tm.one_hot(tm.STOCK), tax)["val"]
                              for _ in range(N_RANDOM)])
            rec["random_p95"] = float(np.percentile(fakes, 95))
            rec["random_pct"] = float((fakes < r12["val"]).mean() * 100)
        out["switch"][n] = rec
        # 2배 균형 (나스닥은 참고)
        b = balanced(m, s, e, fx_all, tax)
        b6040 = balanced(m, s, e, fx_all, tax, 0.60, lev=False)
        out["balanced"][n] = {"window": b["window"], "basis": rec["basis"], "stock_cagr": base["val"], "stock_mdd": base["mdd"],
                              "judged": n in ("S&P500", "QQQ(보정)"), "cagr": b["val"], "mdd": b["mdd"], "rebalances": b["trades"],
                              "cagr_50": balanced(m, s, e, fx_all, tax, 0.50)["val"], "cagr_65": balanced(m, s, e, fx_all, tax, 0.65)["val"],
                              "sharpe": tm.sharpe(b["nav"], b["cash"]), "sharpe_6040": tm.sharpe(b6040["nav"], b6040["cash"]),
                              "cagr_6040": b6040["val"], "mdd_6040": b6040["mdd"]}
    subs = []
    for n, lst in SUBS.items():
        m = mk[n]
        tax = n == "QQQ(보정)"
        for lab, s, e in lst:
            base = stock_hold(m, s, e, fx_all, tax)
            sw = book(m, s, e, switch_decisions(m, s, 12), changed, fx_all, tm.one_hot(tm.STOCK), tax)
            bl = balanced(m, s, e, fx_all, tax)
            subs.append({"구간": n, "하위": lab, "주식 1배": base["val"], "갈아타기": sw["val"], "2배 균형": bl["val"],
                         "갈아타기 판정 대상": n == "QQQ(보정)"})
    out["subs"] = subs
    # 자료 확인: 합성 장기채(DGS10, 만기 15년) vs TLT, 금 선물 GC=F vs GLD — 겹치는 기간 일간 상관·연복리
    q = mk["QQQ(보정)"]["d"]
    syn = tm.bond_returns(asof_series(_sealed_csv("sealed_fred_DGS10.csv"), q))
    tlt = _sealed_yf("TLT").reindex(q).pct_change()
    ok = tlt.notna() & (q > _sealed_yf("TLT").index[0])
    gc = asof_series(_sealed_yf("GC_F", "Close"), q).pct_change()
    gld = _sealed_yf("GLD").reindex(q).pct_change()
    okg = gld.notna() & gc.notna() & (q > _sealed_yf("GLD").index[0])
    yrs = lambda m: (q[m][-1] - q[m][0]).days / 365.25
    out["checks"] = {
        "bond": {"window": f"{q[ok][0].date()} ~ {q[ok][-1].date()}", "corr": float(np.corrcoef(syn[ok], tlt[ok])[0, 1]),
                 "cagr_synth": float((1 + syn[ok]).prod() ** (1 / yrs(ok)) - 1), "cagr_tlt": float((1 + tlt[ok]).prod() ** (1 / yrs(ok)) - 1)},
        "gold": {"window": f"{q[okg][0].date()} ~ {q[okg][-1].date()}", "corr": float(np.corrcoef(gc[okg], gld[okg])[0, 1]),
                 "cagr_gc": float((1 + gc[okg]).prod() ** (1 / yrs(okg)) - 1), "cagr_gld": float((1 + gld[okg]).prod() ** (1 / yrs(okg)) - 1)}}
    if N_RANDOM != 1000:
        print("점검 실행 — 파일 안 씀")
        return
    (OUT / "two_more_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
