"""상승장 대형주 웅덩이 갈아타기 — 3단계 실행 (docs/prereg/puddle_megacap.md, 태그 puddle-megacap-prereg 뒤에만).

봉인 ⓑ: 모든 가격·환율은 scripts.puddle_data에서 2021-12-31까지만 받은 것. 여기서도 한 번 더 자른다.
출력: docs/research/puddle_megacap_result.json, exports/puddle/puddle_trades.xlsx(gitignore)
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import puddle as pu  # noqa: E402
from scripts import puddle_coverage as pc  # noqa: E402
from scripts import puddle_data as pdd  # noqa: E402

CUTOFF = pd.Timestamp("2021-12-31")
SEGMENTS = {"A": ("2000-01-01", "2011-12-31"), "B": ("2012-01-01", "2021-12-31")}
SEED = 20261023
N_RANDOM = 1000
WINDOWS = {  # 표준 보고: 급락·V자 구간(사전 등록 3장), 세전 달러 가치 수익률
    "A": [("닷컴 급락", "2000-03-27", "2002-10-09"), ("2008 급락", "2007-10-31", "2009-03-09"),
          ("2009 V자", "2009-03-09", "2009-09-30")],
    "B": [("2018 4분기 급락", "2018-08-29", "2018-12-24"), ("2020 급락", "2020-02-19", "2020-03-23"),
          ("2020 V자", "2020-03-23", "2020-08-31")],
}
CONCENTRATION_YEARS = {"A": [], "B": [2017, 2020, 2021]}  # 쏠림 구간(실패 지도 카드 2)


def _cut(x):
    out = x[x.index <= CUTOFF]
    assert len(out) == 0 or out.index.max() <= CUTOFF, "봉인 ⓑ 값"
    return out


def load_world():
    """QQQ·종목·환율·명단 → build_inputs 재료."""
    q = _cut(pdd.fetch_price("QQQ"))
    membership = json.loads((pdd.CACHE / "membership.json").read_text())
    members, cols = {}, set()
    for d, mem in membership.items():
        ys = set()
        for t in mem:
            y = pdd.yf_ticker(t)
            if y and pdd.fetch_price(y) is not None:
                ys.add(y)
        members[pd.Timestamp(d)] = ys
        cols |= ys
    cols = sorted(cols)
    frames = {c: _cut(pdd.fetch_price(c)) for c in cols}
    close = pd.DataFrame({c: f["close"] for c, f in frames.items()}).reindex(q.index)
    adj = pd.DataFrame({c: f["adj"] for c, f in frames.items()}).reindex(q.index)
    vol = pd.DataFrame({c: f["volume"] for c, f in frames.items()}).reindex(q.index)
    company = {c: pc.company_of(c) for c in cols}
    fx = _cut(pdd.fetch_fx())
    return q, close, adj, vol, members, company, fx


def seg_inputs(world, seg, p):
    q, close, adj, vol, members, company, fx = world
    a, b = SEGMENTS[seg]
    inp, _ = pu.build_inputs(q["close"], q["adj"], close, adj, vol, members, company, fx, a, b, p)
    return inp


def after_tax_cagr(inp, r):
    days = (inp.dates[-1] - inp.dates[0]).days
    return pu.cagr(100_000_000.0, r.end_after_tax_krw, days)


def window_ret(inp, nav, a, b):
    d = inp.dates
    i, j = d.searchsorted(pd.Timestamp(a)), min(d.searchsorted(pd.Timestamp(b), side="right") - 1, len(d) - 1)
    return float(nav[j] / nav[i] - 1)


def year_ret(inp, nav, y):
    idx = np.where(inp.dates.year == y)[0]
    prev = np.where(inp.dates.year == y - 1)[0]
    start = prev[-1] if len(prev) else idx[0]
    return float(nav[idx[-1]] / nav[start] - 1)


def summarize(inp, r, q_res):
    years = (inp.dates[-1] - inp.dates[0]).days / 365.25
    closed = [t for t in r.trades if t["reason"] != "구간 끝"]
    by_reason = {}
    for reason in pu.REASONS + ("구간 끝",):
        ts = [t for t in r.trades if t["reason"] == reason]
        if ts:
            by_reason[reason] = {"n": len(ts), "avg_excess": float(np.nanmean([t["excess"] for t in ts]))}
    return {
        "cagr_after_tax": after_tax_cagr(inp, r), "mdd": pu.max_drawdown(r.nav),
        "end_after_tax_krw": r.end_after_tax_krw, "avg_stock_weight": float(np.mean(r.stock_w)),
        "fills_per_year": len(r.fills) / years, "positions_per_year": len(r.trades) / years,
        "avg_hold_days": float(np.mean([t["exit_pos"] - t["entry_pos"] for t in closed])) if closed else None,
        "by_reason": by_reason, "costs_krw": r.costs_krw, "taxes_krw": r.taxes_krw,
        "costs_krw_vs_qqq": r.costs_krw - q_res.costs_krw, "taxes_krw_vs_qqq": r.taxes_krw - q_res.taxes_krw,
        "harvests": r.harvests, "n_positions": len(r.trades), "n_fills": len(r.fills),
    }


def export_trades(world_inputs: dict, results: dict, path: Path) -> None:
    rows = []
    for seg, (inp, r) in world_inputs.items():
        for t in r.trades:
            for i, (pp, usd) in enumerate(t["fills"], 1):
                q_ret = inp.qqq_px[t["exit_pos"]] / inp.qqq_px[pp] - 1
                px = pd.Series(inp.px[:, t["k"]]).ffill().to_numpy()
                s_ret = px[t["exit_pos"]] / px[pp] - 1
                rows.append({"구간": seg, "종목": inp.names[t["k"]], "차수": i, "매수일": inp.dates[pp].date(),
                             "매도일": inp.dates[t["exit_pos"]].date(), "매도 이유": t["reason"],
                             "매수 금액($)": round(usd, 2), "수익률(세전·비용 전)": round(s_ret, 4),
                             "QQQ 같은 기간": round(q_ret, 4), "QQQ 대비": round((1 + s_ret) / (1 + q_ret) - 1, 4)})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(path, index=False)


def main() -> None:
    t0 = time.time()
    world = load_world()
    base, robust, proto = pu.Params(), pu.Params(stages=(10, 50, 100)), pu.Params(mode="cash30")
    rng = np.random.default_rng(SEED)
    out, keep = {}, {}
    for seg in SEGMENTS:
        inp = seg_inputs(world, seg, base)
        q_res = pu.simulate(inp, base, no_stocks=True)
        r = pu.simulate(inp, base)
        keep[seg] = (inp, r)
        inp_rb = seg_inputs(world, seg, robust)
        r_rb = pu.simulate(inp_rb, robust)
        r_pr = pu.simulate(inp, proto)
        q_pr = pu.simulate(inp, proto, no_stocks=True)  # 원형 비교용: QQQ 70%·현금 30% 그대로
        rand = []
        for _ in range(N_RANDOM):
            rr = pu.simulate(inp, base, random_days=r.entry_signal_days, rng=rng)
            rand.append(after_tax_cagr(inp, rr))
        rand = np.array(rand)
        s = summarize(inp, r, q_res)
        qc = after_tax_cagr(inp, q_res)
        qm = pu.max_drawdown(q_res.nav)
        tops = [len(v) for v in inp.top.values()]
        out[seg] = {
            "window": [str(inp.dates[0].date()), str(inp.dates[-1].date())],
            "strategy": s, "qqq": {"cagr_after_tax": qc, "mdd": qm, "costs_krw": q_res.costs_krw, "taxes_krw": q_res.taxes_krw},
            "robust_10_50_100": {"cagr_after_tax": after_tax_cagr(inp_rb, r_rb), "mdd": pu.max_drawdown(r_rb.nav)},
            "proto_cash30": {"cagr_after_tax": after_tax_cagr(inp, r_pr), "mdd": pu.max_drawdown(r_pr.nav),
                             "qqq70_cash30_cagr": after_tax_cagr(inp, q_pr)},
            "random": {"n": N_RANDOM, "p95": float(np.nanpercentile(rand, 95)), "median": float(np.nanmedian(rand)),
                       "pct_rank": float((rand < s["cagr_after_tax"]).mean() * 100), "nan": int(np.isnan(rand).sum())},
            "criteria": {"1_beats_qqq": s["cagr_after_tax"] > qc,
                         "2_random_p95": s["cagr_after_tax"] >= float(np.nanpercentile(rand, 95)),
                         "3_mdd_within_2pp": s["mdd"] >= qm - 0.02,
                         "4_robust_beats_qqq": after_tax_cagr(inp_rb, r_rb) > qc},
            "windows": [{"name": nm, "from": a, "to": b, "strategy": window_ret(inp, r.nav, a, b),
                         "qqq": window_ret(inp, q_res.nav, a, b)} for nm, a, b in WINDOWS[seg]],
            "concentration": [{"year": y, "strategy": year_ret(inp, r.nav, y), "qqq": year_ret(inp, q_res.nav, y)}
                              for y in CONCENTRATION_YEARS[seg]],
            "top10_size_min": min(tops), "entry_signal_days": len(r.entry_signal_days),
            "top_names": Counter(inp.names[k] for v in inp.top.values() for k in v).most_common(15),
        }
        print(seg, json.dumps({k: out[seg][k] for k in ("strategy", "qqq", "criteria", "random")}, ensure_ascii=False,
                              default=float)[:1500], flush=True)
    (ROOT / "docs" / "research" / "puddle_megacap_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    export_trades(keep, out, ROOT / "exports" / "puddle" / "puddle_trades.xlsx")
    print(f"끝 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
