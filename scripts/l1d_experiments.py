"""L1d 해외 시장 재현 + 3배 검증 (사전 등록: docs/l1d_plan.md, configs/l1d_preregistration.yaml, tag l1d-prereg).

일회성 리포트 스크립트. 신호·레버리지·장부는 core/l1b.py·core/l1d.py(순수 함수), 데이터는 data/l1d_history.py.
시작할 때 사전 등록 두 파일의 sha256(LF 정규화)을 확인하고 다르면 멈춘다. 2022-01-01 이후 행은 로딩 단계에서
자르고, 하나라도 남으면 멈춘다.

실행: python -u -m scripts.l1d_experiments --check-data   (데이터 검사표만)
      python -u -m scripts.l1d_experiments                (전체)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import l1b, l1d  # noqa: E402
from data import l1d_history as hist  # noqa: E402

PREREG_FILES = {
    "docs/l1d_plan.md": "c240731486f8ee8cc799a6bafd8a5e1b402fc24f896dce41219f0224bf8cd320",
    "configs/l1d_preregistration.yaml": "8a93c1ab00893e475bf082fac4e1efd3058d7df5f41bd1ffaad7c83e44de9caf",
}
OUT_DIR = ROOT / "outputs"
RESULT_MD = ROOT / "docs" / "results" / "l1d.md"
MARKETS = ["JP", "DE", "GB", "HK", "KR"]
MARKET_KO = {"JP": "일본", "DE": "독일", "GB": "영국", "HK": "홍콩", "KR": "한국"}
CANDS = ["G0", "G1", "G2"]
ASSET_KO = {"1x": "1배", "2x": "2배", "3x": "3배", "cash": "국채"}


# ── 사전 등록 확인 ───────────────────────────────────────────────────────────


def normalized_sha256(path: Path) -> str:
    """줄바꿈을 LF로 맞춘 내용의 sha256 (Windows 체크아웃의 CRLF 변환에 흔들리지 않게)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_preregistration() -> dict:
    for rel, expected in PREREG_FILES.items():
        got = normalized_sha256(ROOT / rel)
        if got != expected:
            raise SystemExit(f"[l1d] 사전 등록 파일이 바뀌었습니다: {rel} (기대 {expected[:12]}…, 실제 {got[:12]}…) — 멈춤")
    print("[l1d] 사전 등록 해시 확인 통과", flush=True)
    with open(ROOT / "configs" / "l1d_preregistration.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 데이터 ───────────────────────────────────────────────────────────────────


def load_markets(prereg: dict) -> dict:
    """5개 시장의 지수·금리 연결·수익 표·판정 시작일. 모두 봉인 검사를 거친다."""
    pri = prereg["data"]["rates"]["priority"]
    need = sorted({sid for ids in pri.values() for sid in ids})
    fred = {sid: hist.load_fred_monthly(sid) for sid in need}
    warm = int(prereg["data"]["warmup_rows"])
    out = {}
    for mk in MARKETS:
        close, info = hist.load_index(prereg["markets"][mk]["ticker"])
        months = pd.period_range(close.index[0].to_period("M"), close.index[-1].to_period("M"), freq="M")
        chain = l1d.chain_monthly_rates({sid: fred[sid] for sid in pri[mk]}, pri[mk], months)
        if chain["rate_pct"].isna().any():
            raise SystemExit(f"[l1d] {mk} 금리가 없는 달이 있습니다: {list(chain.index[chain['rate_pct'].isna()])[:5]} — 멈춤")
        rf = l1d.daily_rate(close.index, chain)
        rets = l1d.market_returns(close, rf)
        for name, obj in (("close", close), ("rf", rf), ("returns", rets)):
            l1d.seal(obj, f"{mk}_{name}")
        out[mk] = {"close": close, "info": info, "chain": chain, "rf": rf, "returns": rets,
                   "start": close.index[warm], "end": close.index[-1]}
    return out


def data_check(ctx: dict, prereg: dict) -> tuple[list, list, list, list, list]:
    """데이터 검사표(구현 해석 2~6번). 알려진 값·나쁜 틱 검사를 어기면 멈춘다."""
    series, rates, ticks, checks = [], [], [], []
    for mk in MARKETS:
        m = ctx[mk]
        c = m["close"]
        gaps = pd.Series(c.index[1:] - c.index[:-1], index=c.index[1:])
        big = gaps[gaps > pd.Timedelta(days=10)]
        series.append({
            "시장": f"{MARKET_KO[mk]} {prereg['markets'][mk]['ticker']}", "시작": c.index[0].date(), "끝": c.index[-1].date(),
            "행 수": len(c), "판정 시작(210행 뒤)": m["start"].date(), "뺀 행(NaN/0이하/중복)": f"{m['info']['nan']}/{m['info']['nonpositive']}/{m['info']['duplicates']}",
            "10일 넘는 공백": ", ".join(f"{d.date()}({v.days}일)" for d, v in big.items()) or "없음",
            "2022 이후 행": int((c.index > l1d.SEAL_DATE).sum()),
        })
        for p in l1d.source_periods(m["chain"]):
            rates.append({"시장": MARKET_KO[mk], "시리즈": p["source"], "기간": f"{p['start']} ~ {p['end']}", "개월": p["months"],
                          "대체": "DTB3 대체" if p["source"] == "DTB3" else ""})
        r = c.pct_change()
        allowed = {(a["market"], a["date"]) for a in prereg["data"]["bad_tick_guard"]["allowed"]}
        for d, v in r[r.abs() >= prereg["data"]["bad_tick_guard"]["abs_daily_return_ge"]].items():
            ok = (mk, str(d.date())) in allowed
            ticks.append({"시장": MARKET_KO[mk], "날짜": d.date(), "하루 수익 %": round(v * 100, 2), "알려진 사건": "예" if ok else "아니오"})
    bad = [t for t in ticks if t["알려진 사건"] == "아니오"]

    for kv in prereg["data"]["known_values"]:
        c = ctx[kv["market"]]["close"]
        if kv["check"] == "close":
            got = float(c.loc[kv["date"]])
            ok = abs(got / kv["value"] - 1) * 100 <= kv["tol_pct"]
            checks.append({"항목": f"{MARKET_KO[kv['market']]} {kv['date']} 종가", "계산": round(got, 2), "알려진 값": kv["value"], "통과": ok})
        elif kv["check"] == "daily_return":
            pos = c.index.get_loc(pd.Timestamp(kv["date"]))
            got = (c.iloc[pos] / c.iloc[pos - 1] - 1) * 100
            checks.append({"항목": f"{MARKET_KO[kv['market']]} {kv['date']} 하루 % (전 거래일 {c.index[pos - 1].date()} 대비)", "계산": round(got, 2), "알려진 값": kv["value_pct"], "통과": abs(got - kv["value_pct"]) <= kv["tol_pp"]})
        elif kv["check"] == "month_min_close":
            got = float(c.loc[kv["month"]].min())
            checks.append({"항목": f"{MARKET_KO[kv['market']]} {kv['month']} 최저 종가 ({c.loc[kv['month']].idxmin().date()})", "계산": round(got, 2), "알려진 값": f"{kv['range'][0]}~{kv['range'][1]}", "통과": kv["range"][0] <= got <= kv["range"][1]})
        elif kv["check"] == "two_day_return":
            d1 = c.index.get_loc(pd.Timestamp(kv["dates"][0]))
            d2 = c.index.get_loc(pd.Timestamp(kv["dates"][1]))
            got = (c.iloc[d2] / c.iloc[d1 - 1] - 1) * 100
            checks.append({"항목": f"{MARKET_KO[kv['market']]} {kv['dates'][0]}·{kv['dates'][1]} 이틀 %", "계산": round(got, 2), "알려진 값": kv["value_pct"], "통과": abs(got - kv["value_pct"]) <= kv["tol_pp"]})
        elif kv["check"] == "annual_return":
            y = kv["year"]
            got = (c.loc[str(y)].iloc[-1] / c.loc[str(y - 1)].iloc[-1] - 1) * 100
            checks.append({"항목": f"{MARKET_KO[kv['market']]} {y}년 연간 %", "계산": round(got, 2), "알려진 값": kv["value_pct"], "통과": abs(got - kv["value_pct"]) <= kv["tol_pp"]})
    return series, rates, ticks, checks, bad


# ── 시뮬레이션 ───────────────────────────────────────────────────────────────


def summarize(sim: dict, pre: dict | None, lev_asset: str | None) -> dict:
    v, held = sim["values"], sim["held"]
    dd = l1b.max_drawdown(v)
    rec = l1b.longest_recovery(v)
    yr = l1b.yearly_returns(v)
    worst = yr.idxmin()
    return {
        "posttax": sim["cagr_pct"], "pretax": pre["cagr_pct"] if pre else None, "mdd": dd["mdd_pct"],
        "mdd_span": f"{dd['peak'].date()} → {dd['trough'].date()}", "recovery_rows": rec["rows"],
        "recovery_span": f"{rec['start'].date()} → {rec['end'].date()}" + (" (진행 중)" if rec["open_ended"] else ""),
        "worst_year": f"{worst} {yr.loc[worst]:.1f}%", "switches": sim["switches"], "tax": sum(sim["taxes"].values()),
        "lev_share": float((held == lev_asset).mean() * 100) if lev_asset else 0.0,
        "wiped": bool((v <= 0).any()), "yearly": yr, "sim": sim,
    }


def run(mk: dict, cand: str, sma_days: int = 200, pretax: bool = False) -> dict:
    tgt = l1d.target(cand, mk["close"], sma_days)
    held = l1d.held_after_close(tgt).loc[mk["start"]:mk["end"]]
    sim = l1b.simulate(held, mk["returns"])
    pre = l1b.simulate(held, mk["returns"], apply_tax=False) if pretax else None
    return summarize(sim, pre, l1d.LEV_ASSET.get(cand))


def regression_check(prereg: dict) -> list[dict]:
    """일반화 레버리지 함수(k=2)로 L1b B1 주 설정을 다시 돌려 L1b 기록과 비교(구현 해석 16번). 다르면 멈춘다."""
    from scripts import l1b_experiments as l1bx

    with open(ROOT / "configs" / "l1b_preregistration.yaml", encoding="utf-8") as f:
        pre_b = yaml.safe_load(f)
    ctx_b = l1bx.load_markets(pre_b)
    exp = prereg["regression_check"]["expected"]
    tol = prereg["regression_check"]["tolerance_pp"]
    rows = []
    for mk, start in (("U", "1927-07-01"), ("N", "1972-01-03")):
        m = ctx_b[mk]
        rets = m["returns"].copy()
        rets["2x"] = l1b.synthetic_leveraged_returns(m["r1"], m["rf"], 2.0)
        held = l1b.held_after_close_daily(l1b.target_b1(m["level"], 200)).loc[pd.Timestamp(start):pd.Timestamp("1998-12-31")]
        sim = l1b.simulate(held, rets)
        post, mdd = sim["cagr_pct"], l1b.max_drawdown(sim["values"])["mdd_pct"]
        ok = abs(round(post, 2) - exp[mk]["posttax_pct"]) <= tol and abs(round(mdd, 2) - exp[mk]["mdd_pct"]) <= tol
        rows.append({"시장": mk, "세후 % (재실행)": round(post, 4), "L1b 기록": exp[mk]["posttax_pct"], "MDD % (재실행)": round(mdd, 4),
                     "L1b MDD": exp[mk]["mdd_pct"], "일치": ok})
        if not ok:
            raise SystemExit(f"[l1d] 회귀 확인 실패: {rows[-1]} — 멈춤")
    return rows, ctx_b


def crash_rows(mk_key: str, mk: dict, runs: dict, windows: dict) -> list[dict]:
    out = []
    r_idx = mk["close"].pct_change()
    for label, (ws, we) in windows.items():
        ws, we = pd.Timestamp(ws), pd.Timestamp(we)
        if ws < mk["start"]:
            win = r_idx.loc[ws:we]
            out.append({"시장": MARKET_KO[mk_key], "구간": label, "후보": "-", "판정 구간": "밖(워밍업)",
                        "레버리지 %": None, "국채 %": None, "전환(체결일→자산)": f"지수 하루 최저 {win.min() * 100:.1f}% ({win.idxmin().date()})" if len(win) else "-",
                        "창 손실 %": round(float((mk["close"].loc[:we].iloc[-1] / mk["close"].loc[:ws - pd.Timedelta(days=1)].iloc[-1] - 1) * 100), 1),
                        "창 안 최대 낙폭 %": None, "G0 창 손실 %": None})
            continue
        g0 = runs["G0"]["sim"]["values"]
        g0_loss = (g0.loc[:we].iloc[-1] / g0.iloc[g0.index.searchsorted(ws) - 1] - 1) * 100
        for cand in CANDS:
            v, held = runs[cand]["sim"]["values"], runs[cand]["sim"]["held"]
            pos0 = v.index.searchsorted(ws) - 1
            base = v.iloc[pos0]
            wv, wh = v.loc[ws:we], held.loc[ws:we]
            chg = held.iloc[pos0:].loc[:we]
            prev = chg.shift(1)
            moves = chg[(chg != prev) & prev.notna()]
            lev = l1d.LEV_ASSET.get(cand)
            out.append({
                "시장": MARKET_KO[mk_key], "구간": label, "후보": cand, "판정 구간": "안",
                "레버리지 %": round(float((wh == lev).mean() * 100), 1) if lev else 0.0,
                "국채 %": round(float((wh == "cash").mean() * 100), 1),
                "전환(체결일→자산)": "; ".join(f"{d.date()}→{ASSET_KO[a]}" for d, a in moves.items())[:260] or "없음",
                "창 손실 %": round(float((wv.iloc[-1] / base - 1) * 100), 1) if base > 0 else None,
                "창 안 최대 낙폭 %": round(float(l1b.max_drawdown(v.iloc[pos0:].loc[:we])["mdd_pct"]), 1) if base > 0 else None,
                "G0 창 손실 %": round(float(g0_loss), 1),
            })
    return out


def worst_days(mk_key: str, mk: dict, runs: dict, n: int = 5) -> tuple[list[dict], dict]:
    rets = mk["returns"].loc[mk["start"]:mk["end"]]
    rows = []
    g1v, g2v = runs["G1"]["sim"]["values"], runs["G2"]["sim"]["values"]
    g1c, g2c = g1v.pct_change(), g2v.pct_change()
    for d in rets["1x"].nsmallest(n).index:
        rows.append({
            "시장": MARKET_KO[mk_key], "날짜": d.date(), "지수 %": round(rets.at[d, "1x"] * 100, 2),
            "합성 2배 %": round(rets.at[d, "2x"] * 100, 2), "합성 3배 %": round(rets.at[d, "3x"] * 100, 2),
            "G1 보유": ASSET_KO[runs["G1"]["sim"]["held"].shift(1).get(d, "1x")] if d != rets.index[0] else "-",
            "G1 계좌 %": round(float(g1c.get(d, np.nan)) * 100, 2) if pd.notna(g1c.get(d, np.nan)) else None,
            "G2 보유": ASSET_KO[runs["G2"]["sim"]["held"].shift(1).get(d, "1x")] if d != rets.index[0] else "-",
            "G2 계좌 %": round(float(g2c.get(d, np.nan)) * 100, 2) if pd.notna(g2c.get(d, np.nan)) else None,
        })
    extra = {
        "시장": MARKET_KO[mk_key],
        "합성 3배 하루 ≤ −30% 날 수": int((rets["3x"] <= -0.30).sum()),
        "G2 계좌 하루 −30% 이상 손실 날 수": int((g2c <= -0.30).sum()),
        "G1 계좌 0 도달": runs["G1"]["wiped"], "G2 계좌 0 도달": runs["G2"]["wiped"],
    }
    return rows, extra


def market_verdict(runs: dict, nbs: dict) -> dict:
    out = {}
    for cand in ("G1", "G2"):
        per = {}
        for mk in MARKETS:
            d1 = runs[mk][cand]["posttax"] - runs[mk]["G0"]["posttax"]
            d2 = runs[mk][cand]["mdd"] - runs[mk]["G0"]["mdd"]
            d3 = [x for _, _, x in nbs[mk][cand]]
            ok = {1: d1 >= 2.0, 2: d2 >= -2.0, 3: all(abs(x) <= 1.0 for x in d3)}
            per[mk] = {"ok": ok, "pass": all(ok.values()), "d1": d1, "d2": d2, "d3": d3}
        n_pass = sum(per[mk]["pass"] for mk in MARKETS)
        floor_ok = all(per[mk]["d1"] >= -2.0 for mk in MARKETS)
        out[cand] = {"markets": per, "n_pass": n_pass, "floor_ok": floor_ok, "pass": n_pass >= 4 and floor_ok,
                     "worst_mdd": min(runs[mk][cand]["mdd"] for mk in MARKETS)}
    return out


# ── 참고 보고 (판정 제외) ─────────────────────────────────────────────────────


def reference_us(ctx_b: dict) -> list[dict]:
    rows = []
    for mk, start, label in (("U", "1927-07-01", "미국 시장(프렌치) 1927-07~1998"), ("N", "1972-01-03", "나스닥 ^IXIC 1972~1998")):
        m = ctx_b[mk]
        rets = pd.DataFrame({"1x": m["r1"], "2x": l1b.synthetic_leveraged_returns(m["r1"], m["rf"], 2.0),
                             "3x": l1b.synthetic_leveraged_returns(m["r1"], m["rf"], 3.0), "cash": m["rf"]})
        mkd = {"close": m["level"], "returns": rets, "start": pd.Timestamp(start), "end": pd.Timestamp("1998-12-31")}
        res = {c: run(mkd, c, pretax=True) for c in CANDS}
        for c in CANDS:
            r = res[c]
            rows.append({"기간": label, "후보": c, "세후 %": r["posttax"], "세전 %": r["pretax"], "MDD %": r["mdd"],
                         "G0 대비 %p": r["posttax"] - res["G0"]["posttax"], "레버리지 기간 %": r["lev_share"], "계좌 0 도달": r["wiped"]})
    return rows


def reference_qqq() -> list[dict]:
    from dataclasses import replace

    from engine import backtest as bt
    from engine import portfolio as pf
    from scripts import l1_experiments as l1x

    cfg = bt.load_config()
    ctx = l1x.load_all(cfg, {"periods": {"full": {"start": "1999-03-10", "end": "2021-12-31"}}})
    data, start, end = ctx["data"], ctx["start"], ctx["end"]
    td = data.qqq_df.index
    r = data.qqq_df["close"].pct_change().fillna(0.0)
    rate = data.reserve_daily_rate.reindex(td).fillna(0.0)
    lev3 = l1b.synthetic_leveraged_returns(r, rate, 3.0)
    level = 100.0 * (1.0 + lev3).cumprod()
    data3 = replace(data, qld_df=pd.DataFrame({"open": level, "high": level, "low": level, "close": level}, index=td),
                    qld_dividends=pd.Series(dtype=float))
    base = l1x.run_scenario(ctx, cfg, "L0", start, end)
    rows = [{"후보": "L0 (QQQM 보유, P6 P0)", "세후a %": base["posttax_a_pct"], "MDD %": base["mdd_pct"], "전환": 0, "3배 기간 %": 0.0}]
    for cand, dat, lev_name in (("G1 (QLD 2배, L1 L2와 같음)", data, "2x"), ("G2 (합성 3배)", data3, "3x")):
        tgt = l1d.target("G1" if lev_name == "2x" else "G2", data.sma_source_df["close"]).reindex(td)
        if tgt.isna().any():
            raise SystemExit("[l1d] 참고 보고 QQQ 신호 미산출일 — 멈춤")
        w = pd.DataFrame([(0.0, 1.0, 0.0) if a == lev_name else (0.0, 0.0, 1.0) for a in tgt], index=td, columns=["core", "qld", "reserve"])
        w["regime"] = ["공격" if a == lev_name else "방어" for a in tgt]
        res = pf.simulate_target_weights(dat, cfg, start, end, w)
        eq = [x for x in res.equity_rows if x["total_krw"] is not None]
        m = bt.compute_equity_metrics(eq)
        a = pf.compute_posttax_a(res, dat, cfg, end)
        rows.append({"후보": cand, "세후a %": a.get("cagr_liquidated_pct"), "MDD %": m.get("mdd_pct"), "전환": res.trade_count,
                     "3배 기간 %": round(float((tgt.loc[start:end] == lev_name).mean() * 100), 1) if lev_name == "3x" else None})
    return rows


# ── 실행·출력 ───────────────────────────────────────────────────────────────


def run_all(ctx: dict, prereg: dict, series, rates, ticks, checks) -> None:
    print("[l1d] 회귀 확인(k=2 = L1b) ...", flush=True)
    regression, ctx_b = regression_check(prereg)
    print_table(regression, "회귀 확인")
    runs, nbs = {}, {}
    for mk in MARKETS:
        print(f"[l1d] {mk} {ctx[mk]['start'].date()} ~ {ctx[mk]['end'].date()}", flush=True)
        runs[mk] = {c: run(ctx[mk], c, pretax=True) for c in CANDS}
        nbs[mk] = {}
        for c in ("G1", "G2"):
            nbs[mk][c] = []
            for n in prereg["candidates"]["neighbors"]["sma_days"]:
                if n != 200:
                    r = run(ctx[mk], c, n)
                    nbs[mk][c].append((f"sma_days={n}", r, r["posttax"] - runs[mk][c]["posttax"]))
    verdict = market_verdict(runs, nbs)
    crash = [row for mk in MARKETS for row in crash_rows(mk, ctx[mk], runs[mk], prereg["reporting"]["crash_windows"][mk])]
    worst, worst_extra = [], []
    for mk in MARKETS:
        w, e = worst_days(mk, ctx[mk], runs[mk])
        worst += w
        worst_extra.append(e)
    print("[l1d] 참고 보고 ...", flush=True)
    ref_us = reference_us(ctx_b)
    try:
        ref_qqq, ref_err = reference_qqq(), None
    except Exception as e:  # 참고 보고 실패는 판정과 무관 — 숨기지 않고 적는다
        ref_qqq, ref_err = [], f"{type(e).__name__}: {e}"
    write_outputs(ctx, prereg, series, rates, ticks, checks, regression, runs, nbs, verdict, crash, worst, worst_extra, ref_us, ref_qqq, ref_err)


def _f(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return "예" if x else "아니오"
    if isinstance(x, (float, np.floating)):
        return f"{x:,.{nd}f}"
    return str(x)


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "(없음)\n"
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(_f(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def make_plots(ctx: dict, runs: dict) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"G0": "#555555", "G1": "#1f77b4", "G2": "#d62728"}
    paths = []
    for kind in ("equity_log", "drawdown"):
        fig, axes = plt.subplots(len(MARKETS), 1, figsize=(13, 3.4 * len(MARKETS)))
        for ax, mk in zip(axes, MARKETS):
            for c in CANDS:
                v = runs[mk][c]["sim"]["values"]
                if kind == "equity_log":
                    ax.plot(v.index, v.clip(lower=1.0).values, color=colors[c], lw=1, label=f"{c} 세후 {runs[mk][c]['posttax']:.2f}%")
                else:
                    ax.plot(v.index, (v / v.cummax() - 1) * 100, color=colors[c], lw=0.8, label=f"{c} MDD {runs[mk][c]['mdd']:.1f}%")
            if kind == "equity_log":
                ax.set_yscale("log")
            ax.set_title(f"{MARKET_KO[mk]} — {'자산 곡선(로그, 세후 장부, 시작 100,000 현지 통화)' if kind == 'equity_log' else '낙폭 %'}")
            ax.legend(loc="upper left" if kind == "equity_log" else "lower left", fontsize=8)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        path = OUT_DIR / f"l1d_{kind}.png"
        fig.savefig(path, dpi=105)
        plt.close(fig)
        paths.append(path)
    return paths


def write_outputs(ctx, prereg, series, rates, ticks, checks, regression, runs, nbs, verdict, crash, worst, worst_extra, ref_us, ref_qqq, ref_err) -> None:
    import subprocess

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    tag = subprocess.run(["git", "rev-parse", "l1d-prereg"], capture_output=True, text=True, cwd=ROOT).stdout.strip()

    summary = []
    for mk in MARKETS:
        for c in CANDS:
            r = runs[mk][c]
            summary.append({"시장": MARKET_KO[mk], "후보": c, "세후 %": r["posttax"], "세전 %": r["pretax"],
                            "G0 대비 세후 %p": r["posttax"] - runs[mk]["G0"]["posttax"], "MDD %": r["mdd"], "MDD 고점→저점": r["mdd_span"],
                            "최장 회복(행)": r["recovery_rows"], "회복 구간": r["recovery_span"], "최악의 해": r["worst_year"],
                            "전환": r["switches"], "세금 합계": r["tax"], "레버리지 기간 %": r["lev_share"], "계좌 0 도달": r["wiped"]})
    nb_rows = []
    for mk in MARKETS:
        for c in ("G1", "G2"):
            nb_rows.append({"시장": MARKET_KO[mk], "후보": c, "설정": "200일(주)", "세후 %": runs[mk][c]["posttax"], "MDD %": runs[mk][c]["mdd"], "주 설정 대비 %p": 0.0})
            nb_rows += [{"시장": MARKET_KO[mk], "후보": c, "설정": lb.replace("sma_days=", "") + "일", "세후 %": r["posttax"], "MDD %": r["mdd"], "주 설정 대비 %p": d} for lb, r, d in nbs[mk][c]]
    mv_rows = []
    for c in ("G1", "G2"):
        for mk in MARKETS:
            p = verdict[c]["markets"][mk]
            mark = lambda k: "✓" if p["ok"][k] else "✗"  # noqa: E731
            mv_rows.append({"후보": c, "시장": MARKET_KO[mk], "1 세후 +2%p": f"{mark(1)} {p['d1']:+.2f}", "2 MDD": f"{mark(2)} {p['d2']:+.2f}",
                            "3 주변값": f"{mark(3)} {[round(float(x), 2) for x in p['d3']]}", "통과": "통과" if p["pass"] else "불통과"})
    pd.DataFrame(summary).to_csv(OUT_DIR / "l1d_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(nb_rows).to_csv(OUT_DIR / "l1d_neighbors.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mv_rows).to_csv(OUT_DIR / "l1d_verdict.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(crash).to_csv(OUT_DIR / "l1d_crash.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(worst).to_csv(OUT_DIR / "l1d_worst_days.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rates).to_csv(OUT_DIR / "l1d_rate_sources.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ref_us + [{"기간": "QQQ L1 엔진", **r} for r in ref_qqq]).to_csv(OUT_DIR / "l1d_reference.csv", index=False, encoding="utf-8-sig")
    yearly = pd.DataFrame({f"{mk}_{c}": runs[mk][c]["yearly"] for mk in MARKETS for c in CANDS}).round(2)
    yearly.to_csv(OUT_DIR / "l1d_yearly.csv", encoding="utf-8-sig")
    eq = {}
    for mk in MARKETS:
        for c in CANDS:
            eq[f"{mk}_{c}_value"] = runs[mk][c]["sim"]["values"]
            eq[f"{mk}_{c}_held"] = runs[mk][c]["sim"]["held"]
    pd.DataFrame(eq).to_csv(OUT_DIR / "l1d_equity_daily.csv", encoding="utf-8-sig")
    plots = make_plots(ctx, runs)

    vlines = []
    for c in ("G1", "G2"):
        v = verdict[c]
        failed_floor = [MARKET_KO[mk] for mk in MARKETS if v["markets"][mk]["d1"] < -2.0]
        vlines.append(f"- **{c}: {'합격' if v['pass'] else '불합격'}** — 통과 시장 {v['n_pass']}/5 (기준 4 이상) · "
                      f"모든 시장 세후 −2%p 하한 {'✓' if v['floor_ok'] else '✗ (' + ', '.join(failed_floor) + ')'} · 시장 MDD 최악값 {v['worst_mdd']:.2f}%")
    g1, g2 = verdict["G1"]["pass"], verdict["G2"]["pass"]
    if g1 and g2:
        pick = "G1" if verdict["G1"]["worst_mdd"] > verdict["G2"]["worst_mdd"] else "G2"
        decision = f"**{pick} 채택** (G1·G2 모두 합격, 시장 MDD 최악값이 더 얕은 쪽)"
    elif g2:
        decision = "**G2만 합격 → 참고 보고(미국 3배)를 함께 보고 봉인 개봉 후보로 기록**"
    elif g1:
        decision = "**G1 합격(G2 불합격)** — B1(200일선 2배 전환)이 해외 5개 시장에서도 재현"
    else:
        decision = "**G1·G2 모두 불합격 — 그대로 기록하고 같은 기간을 다시 판정에 쓰지 않는다.**"

    ydf = yearly.copy()
    ydf.insert(0, "연도", ydf.index)
    md = f"""# L1d 해외 시장 재현 + 3배 검증 — 결과 (L1d 시도 1)

- 사전 등록: `docs/l1d_plan.md`, `configs/l1d_preregistration.yaml` (tag `l1d-prereg` = `{tag}`), 시작 시 sha256 확인 통과
  - `docs/l1d_plan.md` `{PREREG_FILES['docs/l1d_plan.md']}`
  - `configs/l1d_preregistration.yaml` `{PREREG_FILES['configs/l1d_preregistration.yaml']}`
- 실행 코드 커밋: `{head}` · 봉인 2021-12-31(2022-01-01 이후 행 0개) · 통화 = 각 시장 현지 통화, 시작 100,000
- **가상 상품 주의**: 2배·3배는 일간 k × 지수 − (k−1) × 단기 금리 − 0.95%/252, 매일 재조정한 가상 합성 상품이다. 하루 −100% 이하면 0으로 고정.
- **체결 규칙**: 신호 = 그날 종가, 체결 = 다음 거래일 종가. 가격 지수라 배당 미포함(DAX만 배당 재투자 지수).

## 판정

{chr(10).join(vlines)}

결론: {decision}

시장별 통과:

{md_table(mv_rows)}
## 회귀 확인 (일반화 레버리지 k=2 = L1b)

{md_table(regression)}
## 데이터 검사표

{md_table(series)}
단기 금리 출처(시장·기간 — 같은 나라 OECD/FRED 시리즈 먼저, 홍콩만 DTB3 대체):

{md_table(rates)}
하루 |수익| 25% 이상:

{md_table(ticks)}
알려진 값 대조:

{md_table(checks)}
## 시장별 후보 (판정 구간, 세후 장부)

{md_table(summary)}
## 주변값 (전 기간 세후, 이동평균만 바꿈, 안정 = ±1%p 이내)

{md_table(nb_rows)}
## 폭락 구간 대응 (전 기간 시뮬레이션, 창 손실 = 창 시작 전날 종가 대비 창 끝, 전환일 = 체결일)

{md_table(crash)}
## 하루 최대 손실 상위 5일 (판정 구간, 지수 하루 수익 기준)

{md_table(worst)}
{md_table(worst_extra)}
## 참고 보고 — 미국 3배 (판정에 쓰지 않음)

L1b 데이터·장부(달러, 다음 날 종가 체결):

{md_table(ref_us)}
QQQ 1999-03-10~2021-12-31, L1 엔진(다음 날 **시가** 체결, 원화 4,000만 원, 250만 원 공제·5월 납부). 3배는 QQQ × 3 − 2 × DTB3 − 0.95%/252 합성:

{md_table(ref_qqq) if ref_qqq else f"실행 실패: {ref_err}"}
## 연도별 수익 (세후 장부 평가액, %)

{md_table(ydf.to_dict("records"))}
## 결과를 본 뒤 바꾼 점

- 없음 (사전 등록과 구현 해석 그대로 1회 실행).

## 한계

- 2배·3배는 가상 합성 상품(실제 상품 없음). 차입 비용 = (k−1) × 현지 단기 금리, 보수 0.95%/252.
- 가격 지수(DAX 제외)라 배당 미포함 — G0·G1·G2 모두에 같지만, 국채로 피한 기간은 배당을 놓친 효과가 없어 레버리지 후보에 약간 유리할 수 있다.
- 현지 금리: 같은 나라 다른 시리즈로 이어 붙임(일본 1979-04 전 중앙은행 금리, 1979-05~2002-03 3개월 CD). 홍콩은 FRED에 단기 금리가 없어 전 기간 미국 DTB3(달러 페그 근거). 음수 금리 그대로.
- 다음 거래일 종가 체결 — 하루짜리 폭락(1987-10 등)은 피할 수 없다. 세금은 현지 통화 근사(나라별 실제 세제 아님).
- 판정 구간은 각 시리즈 첫 210행을 뺀 뒤부터(홍콩 1987-10 폭락은 워밍업 안).

## 그래프

{chr(10).join(f'- `{p.relative_to(ROOT).as_posix()}`' for p in plots)}
"""
    RESULT_MD.write_text(md, encoding="utf-8")
    print(f"[l1d] 결과: {RESULT_MD.relative_to(ROOT)}", flush=True)
    print("\n".join(vlines))
    print("결론:", decision)


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n## {title}")
    if not rows:
        print("(없음)")
        return
    cols = list(rows[0])
    print("| " + " | ".join(cols) + " |")
    print("| " + " | ".join("---" for _ in cols) + " |")
    for r in rows:
        print("| " + " | ".join("-" if r[c] is None else str(r[c]) for c in cols) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-data", action="store_true")
    args = ap.parse_args()
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=ROOT / ".env")  # FRED_API_KEY (값은 출력하지 않는다)
    prereg = verify_preregistration()
    ctx = load_markets(prereg)
    series, rates, ticks, checks, bad = data_check(ctx, prereg)
    print_table(series, "데이터 검사표 — 지수")
    print_table(rates, "데이터 검사표 — 단기 금리 출처(시장·기간)")
    print_table(ticks, "데이터 검사표 — 하루 |수익| 25% 이상")
    print_table(checks, "데이터 검사표 — 알려진 값 대조")
    if bad:
        raise SystemExit(f"[l1d] 알려진 사건이 아닌 25% 이상 하루 변동 {len(bad)}건 — 결과 계산 전에 멈춤")
    if not all(c["통과"] for c in checks):
        raise SystemExit("[l1d] 알려진 값 대조 실패 — 결과 계산 전에 멈춤")
    if args.check_data:
        return
    run_all(ctx, prereg, series, rates, ticks, checks)


if __name__ == "__main__":
    main()
