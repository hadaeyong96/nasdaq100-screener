"""L1b 새 기간 국면 검증 (사전 등록: docs/l1b_plan.md, configs/l1b_preregistration.yaml, tag l1b-prereg).

일회성 리포트 스크립트. 신호·합성 2배·장부는 core/l1b.py(순수 함수), 데이터는 data/l1b_history.py.
시작할 때 사전 등록 두 파일의 sha256(LF 정규화)을 확인하고 다르면 멈춘다. 1999-01-01 이후 행은 로딩 단계에서
자르고, 하나라도 남으면 멈춘다.

실행: python -u -m scripts.l1b_experiments --check-data   (데이터 검사표만)
      python -u -m scripts.l1b_experiments                (전체)
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

from core import l1b  # noqa: E402
from data import l1b_history as hist  # noqa: E402

PREREG_FILES = {
    "docs/l1b_plan.md": "dd8b8cd6fd17b3f0e0a27a5fdc035ad48a6b3e0331b41e14c3e8d53cf3c7fa71",
    "configs/l1b_preregistration.yaml": "091dbc7c73cf0cd33aefa5a25d966e84f82be8210ebdaa7570c9f2588a0f3816",
}
OUT_DIR = ROOT / "outputs"
RESULT_MD = ROOT / "docs" / "results" / "l1b.md"
DAILY = ["B0", "B1", "B2", "B3"]
MONTHLY = ["M0", "M1", "M2", "M3"]
MONTHLY_DEFENSE = {"M1": l1b.ASSET_CASH, "M2": l1b.ASSET_HIDIV, "M3": l1b.ASSET_1X}
ASSET_KO = {"1x": "1배", "2x": "2배", "cash": "국채", "hidiv": "고배당"}


# ── 사전 등록 확인 ───────────────────────────────────────────────────────────


def normalized_sha256(path: Path) -> str:
    """줄바꿈을 LF로 맞춘 내용의 sha256 (Windows 체크아웃의 CRLF 변환에 흔들리지 않게)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_preregistration() -> dict:
    for rel, expected in PREREG_FILES.items():
        got = normalized_sha256(ROOT / rel)
        if got != expected:
            raise SystemExit(f"[l1b] 사전 등록 파일이 바뀌었습니다: {rel} (기대 {expected[:12]}…, 실제 {got[:12]}…) — 멈춤")
    print("[l1b] 사전 등록 해시 확인 통과", flush=True)
    with open(ROOT / "configs" / "l1b_preregistration.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 데이터 ───────────────────────────────────────────────────────────────────


def load_markets(prereg: dict) -> dict:
    """N·U 일별(지수·수익·2배·RF)과 U 월별, Hi 30을 만든다. 전부 봉인 검사를 거친다."""
    fr = hist.load_french_daily()
    hi30 = hist.load_hi30_monthly()
    ixic = hist.load_ixic()

    r_u = fr["mkt_rf"] + fr["rf"]
    u = {
        "level": l1b.level_from_returns(r_u), "r1": r_u, "rf": fr["rf"],
        "r2": l1b.synthetic_2x_returns(r_u, fr["rf"]),
    }
    r_n = ixic.pct_change().fillna(0.0)  # 첫 행(워밍업)의 수익은 쓰지 않는다
    rf_n = l1b.align_rf(fr["rf"], ixic.index)
    n = {"level": ixic, "r1": r_n, "rf": rf_n, "r2": l1b.synthetic_2x_returns(r_n, rf_n)}

    # N 판정 시작: 1972-01-03 이전 행이 210개 미만이면 210일선 첫 산출일로 늦춘다(구현 해석 3번)
    n_start = pd.Timestamp(prereg["data"]["N"]["verdict_start"])
    pre_rows = int((ixic.index < n_start).sum())
    n_start_eff = n_start if pre_rows >= 210 else ixic.index[209]
    n["start"], n["pre_rows"] = n_start_eff, pre_rows
    u["start"] = pd.Timestamp(prereg["data"]["U"]["verdict_start"])

    for mk in (u, n):
        mk["returns"] = pd.DataFrame({"1x": mk["r1"], "2x": mk["r2"], "cash": mk["rf"]})
        for k in ("level", "r1", "r2", "rf", "returns"):
            l1b.seal(mk[k], k)

    # U 월별
    m_ret = pd.DataFrame({
        "1x": l1b.compound_by_month(r_u), "2x": l1b.compound_by_month(u["r2"]), "cash": l1b.compound_by_month(fr["rf"]),
    })
    hi = hi30.copy()
    hi.index = hi.index.to_period("M")
    m_ret["hidiv"] = hi.reindex(m_ret.index.to_period("M")).to_numpy()
    m_level = l1b.month_end_levels(u["level"])
    l1b.seal(m_ret, "monthly_returns")
    return {"N": n, "U": u, "fr": fr, "hi30": hi30, "ixic": ixic, "m_ret": m_ret, "m_level": m_level}


def data_check(ctx: dict, prereg: dict) -> list[dict]:
    """데이터 검사표(구현 해석 8번). 알려진 값과 ±tolerance 밖이면 멈춘다."""
    fr, hi30, ixic = ctx["fr"], ctx["hi30"], ctx["ixic"]
    tol = prereg["data"]["checks"]["tolerance_pp"]
    rows = []

    def gaps(idx):
        d = pd.Series(idx[1:] - idx[:-1], index=idx[1:])
        big = d[d > pd.Timedelta(days=5)]
        return ", ".join(f"{i.date()}({v.days}일)" for i, v in big.items()) or "없음"

    rows.append({"시리즈": "프렌치 일별 Mkt-RF·RF (U, 국채)", "시작": fr.index[0].date(), "끝": fr.index[-1].date(),
                 "행 수": len(fr), "결측(NaN)": int(fr.isna().sum().sum()), "5일 넘는 공백": gaps(fr.index)})
    rows.append({"시리즈": "^IXIC 종가 (N)", "시작": ixic.index[0].date(), "끝": ixic.index[-1].date(),
                 "행 수": len(ixic), "결측(NaN)": int(ixic.isna().sum()), "5일 넘는 공백": gaps(ixic.index)})
    hv = hi30.loc["1927-07-01":]
    rows.append({"시리즈": "프렌치 D/P VW Hi 30 (월별)", "시작": hv.index[0].strftime("%Y-%m"), "끝": hv.index[-1].strftime("%Y-%m"),
                 "행 수": len(hv), "결측(NaN)": int(hv.isna().sum()), "5일 넘는 공백": "해당 없음(월별)"})
    if hv.isna().any():
        raise SystemExit(f"[l1b] Hi 30 판정 구간에 결측이 있습니다: {list(hv[hv.isna()].index)} — 멈춤")
    if fr.isna().any().any() or ixic.isna().any():
        raise SystemExit("[l1b] 일별 데이터에 결측이 있습니다 — 멈춤")

    r_u = fr["mkt_rf"] + fr["rf"]
    ann = (1 + r_u).groupby(r_u.index.year).prod() - 1
    checks = []
    for y, known in prereg["data"]["checks"]["french_mkt_annual"].items():
        got = float(ann.loc[int(y)] * 100)
        checks.append({"항목": f"프렌치 Mkt 총수익 {y}년", "계산 %": round(got, 2), "알려진 값 %": known, "차이 %p": round(got - known, 2)})
    for d, known in prereg["data"]["checks"]["ixic_daily"].items():
        ts = pd.Timestamp(d)
        pos = ixic.index.get_loc(ts)
        got = float((ixic.iloc[pos] / ixic.iloc[pos - 1] - 1) * 100)
        checks.append({"항목": f"^IXIC {d} 하루", "계산 %": round(got, 2), "알려진 값 %": known, "차이 %p": round(got - known, 2)})
    ts = pd.Timestamp("1987-10-19")
    checks.append({"항목": "참고: 프렌치 Mkt 1987-10-19 하루", "계산 %": round(float(r_u.loc[ts] * 100), 2), "알려진 값 %": None, "차이 %p": None})
    for c in checks:
        if c["알려진 값 %"] is not None and abs(c["차이 %p"]) > tol:
            raise SystemExit(f"[l1b] 데이터 검사 실패: {c} — 멈춤")

    n_days, f_days = ctx["ixic"].index, fr.index
    n_only = n_days[(n_days >= "1971-01-01") & ~n_days.isin(f_days)]
    f_only = f_days[(f_days >= n_days[0]) & ~f_days.isin(n_days)]
    extra = [
        {"항목": "^IXIC 1972-01-03 이전 워밍업 행", "값": ctx["N"]["pre_rows"], "비고": f"N 판정 시작 {ctx['N']['start'].date()}"},
        {"항목": "^IXIC에만 있는 날(프렌치 없음)", "값": len(n_only), "비고": ", ".join(str(d.date()) for d in n_only[:8])},
        {"항목": "프렌치에만 있는 날(^IXIC 없음, ^IXIC 시작 이후)", "값": len(f_only), "비고": ", ".join(str(d.date()) for d in f_only[:8])},
        {"항목": "U 판정 시작 전 워밍업 행", "값": int((fr.index < ctx["U"]["start"]).sum()), "비고": "1926-07-01~1927-06-30"},
        {"항목": "U 1952년 이전 토요일 행", "값": int(((fr.index.dayofweek == 5) & (fr.index.year < 1953)).sum()), "비고": "행당 보수 /252 적용"},
        {"항목": "봉인 검사(1999-01-01 이후 행)", "값": 0, "비고": "모든 시리즈 통과"},
    ]
    return rows, checks, extra


# ── 시뮬레이션 묶음 ─────────────────────────────────────────────────────────


def summarize(sim: dict, pre: dict | None = None) -> dict:
    """장부 결과 → 보고 지표 dict."""
    v, held = sim["values"], sim["held"]
    dd = l1b.max_drawdown(v)
    rec = l1b.longest_recovery(v)
    yr = l1b.yearly_returns(v)
    worst = yr.idxmin()
    return {
        "posttax": sim["cagr_pct"], "pretax": pre["cagr_pct"] if pre else None,
        "mdd": dd["mdd_pct"], "mdd_peak": dd["peak"].date(), "mdd_trough": dd["trough"].date(),
        "recovery_rows": rec["rows"], "recovery_span": f"{rec['start'].date()} → {rec['end'].date()}",
        "recovery_open": rec["open_ended"], "worst_year": f"{worst} {yr.loc[worst]:.1f}%",
        "switches": sim["switches"], "tax": sum(sim["taxes"].values()),
        "lev2_share": float((held == l1b.ASSET_2X).mean() * 100), "yearly": yr, "sim": sim,
    }


def run_daily(mk: dict, cand: str, start, end, pretax: bool = False, **params) -> dict:
    target = l1b.daily_target(cand, mk["level"], **params)
    held = l1b.held_after_close_daily(target).loc[pd.Timestamp(start):pd.Timestamp(end)]
    sim = l1b.simulate(held, mk["returns"])
    pre = l1b.simulate(held, mk["returns"], apply_tax=False) if pretax else None
    return summarize(sim, pre)


def run_monthly(ctx: dict, cand: str, start, end, avg_months: int = 10, pretax: bool = False) -> dict:
    m_level, m_ret = ctx["m_level"], ctx["m_ret"]
    if cand == "M0":
        target = pd.Series(l1b.ASSET_1X, index=m_level.index, dtype=object)
    else:
        target = l1b.target_monthly(m_level, avg_months, MONTHLY_DEFENSE[cand])
    # 1927-06 말 종가에 사서 1927-07부터 수익(구현 해석 15·16번) — 시작 행 = start 직전 월말
    pos = m_level.index.searchsorted(pd.Timestamp(start)) - 1
    held = target.iloc[pos:].loc[:pd.Timestamp(end)]
    sim = l1b.simulate(held, m_ret)
    pre = l1b.simulate(held, m_ret, apply_tax=False) if pretax else None
    return summarize(sim, pre)


PRIMARY = {"B0": {}, "B1": {"sma_days": 200}, "B2": {"sma_days": 200, "crash_pct": -30.0},
           "B3": {"sma_days": 200, "confirm_days": 3}}


def neighbor_params(cand: str, nb: dict) -> list[tuple[str, dict]]:
    """한 번에 하나만 바꾼 주변값 목록(주 설정 제외)."""
    base = PRIMARY[cand]
    out = []
    for v in nb["sma_days"]:
        if v != 200:
            out.append((f"sma_days={v}", {**base, "sma_days": v}))
    if cand == "B2":
        out += [(f"crash_pct={v}", {**base, "crash_pct": float(v)}) for v in nb["crash_pct"] if v != -30]
    if cand == "B3":
        out += [(f"confirm_days={v}", {**base, "confirm_days": v}) for v in nb["confirm_days"] if v != 3]
    return out


def crash_rows(market: str, runs: dict, windows: dict) -> list[dict]:
    """폭락 창 대응표(구현 해석 33번)."""
    out = []
    for label, (ws, we) in windows.items():
        ws, we = pd.Timestamp(ws), pd.Timestamp(we)
        b0v = runs["B0"]["sim"]["values"]
        for cand in DAILY:
            v, held = runs[cand]["sim"]["values"], runs[cand]["sim"]["held"]
            pos0 = v.index.searchsorted(ws) - 1
            base = v.iloc[pos0]
            win_v, win_h = v.loc[ws:we], held.loc[ws:we]
            win_full = v.iloc[pos0:].loc[:we]
            chg = held.iloc[pos0:].loc[:we]
            prev = chg.shift(1)
            moves = chg[(chg != prev) & prev.notna()]
            b0_loss = (b0v.loc[:we].iloc[-1] / b0v.iloc[b0v.index.searchsorted(ws) - 1] - 1) * 100
            out.append({
                "시장": market, "구간": label, "후보": cand,
                "창 시작 보유": ASSET_KO[held.iloc[pos0]],
                "2배 %": round(float((win_h == "2x").mean() * 100), 1),
                "1배 %": round(float((win_h == "1x").mean() * 100), 1),
                "국채 %": round(float((win_h == "cash").mean() * 100), 1),
                "전환(체결일→자산)": "; ".join(f"{d.date()}→{ASSET_KO[a]}" for d, a in moves.items())[:300] or "없음",
                "창 손실 %": round(float((win_v.iloc[-1] / base - 1) * 100), 1),
                "창 안 최대 낙폭 %": round(float(l1b.max_drawdown(win_full)["mdd_pct"]), 1),
                "B0 창 손실 %": round(float(b0_loss), 1),
            })
    return out


def daily_verdict(full: dict, subs: dict, nbs: dict, prereg: dict) -> dict:
    """판정 1~4 (구현 해석 27~30번). full[mk][cand], subs[mk][name][cand], nbs[mk][cand] = [(label, posttax, diff)]."""
    need = prereg["periods"]["majority"]
    out = {}
    for cand in ("B1", "B2", "B3"):
        c1 = {mk: full[mk][cand]["posttax"] - full[mk]["B0"]["posttax"] for mk in ("N", "U")}
        c2 = {mk: full[mk][cand]["mdd"] - full[mk]["B0"]["mdd"] for mk in ("N", "U")}
        c3 = {mk: sum(subs[mk][s][cand]["posttax"] > subs[mk][s]["B0"]["posttax"] for s in subs[mk]) for mk in ("N", "U")}
        c4 = {mk: [d for _, _, d in nbs[mk][cand]] for mk in ("N", "U")}
        ok = {
            1: all(x >= 2.0 for x in c1.values()),
            2: all(x >= -2.0 for x in c2.values()),
            3: all(c3[mk] >= need[mk] for mk in ("N", "U")),
            4: all(abs(d) <= 1.0 for mk in c4 for d in c4[mk]),
        }
        out[cand] = {"pass": all(ok.values()), "ok": ok, "c1": c1, "c2": c2, "c3": c3, "c4": c4}
    return out


def monthly_verdict(mfull: dict, mnb: dict) -> dict:
    out = {}
    for cand in ("M1", "M2", "M3"):
        c1 = mfull[cand]["posttax"] - mfull["M0"]["posttax"]
        c2 = mfull[cand]["mdd"] - mfull["M0"]["mdd"]
        c4 = [d for _, _, d in mnb[cand]]
        ok = {1: c1 >= 2.0, 2: c2 >= -2.0, 4: all(abs(d) <= 1.0 for d in c4)}
        out[cand] = {"pass": all(ok.values()), "ok": ok, "c1": c1, "c2": c2, "c4": c4}
    return out


# ── 참고 보고: L1 엔진으로 QQQ 1999~2021 (판정 제외) ───────────────────────────


def reference_l1_engine() -> list[dict]:
    """B2·B3 주 설정을 L1 엔진·세금 그대로 QQQ 1999-03-10~2021-12-31에 돌린다(구현 해석 34번)."""
    from engine import backtest as bt
    from engine import portfolio as pf
    from scripts import l1_experiments as l1x

    cfg = bt.load_config()
    ctx = l1x.load_all(cfg, {"periods": {"full": {"start": "1999-03-10", "end": "2021-12-31"}}})
    data, start, end = ctx["data"], ctx["start"], ctx["end"]
    td = data.qqq_df.index
    sig = data.sma_source_df["close"]
    rows = []
    base = l1x.run_scenario(ctx, cfg, "L0", start, end)
    rows.append({"후보": "L0 (QQQM 보유, P6 P0)", "세후a %": base["posttax_a_pct"], "세후b %": base["posttax_b_pct"],
                 "MDD %": base["mdd_pct"], "전환": base["switches"], "2배 기간 %": 0.0})
    mapping = {"1x": (1.0, 0.0, 0.0), "2x": (0.0, 1.0, 0.0), "cash": (0.0, 0.0, 1.0)}
    for cand in ("B2", "B3"):
        tgt = l1b.daily_target(cand, sig, **PRIMARY[cand]).reindex(td)
        if tgt.isna().any():
            raise SystemExit(f"[l1b] 참고 보고 {cand} 신호 미산출일 있음 — 멈춤")
        w = pd.DataFrame([mapping[a] for a in tgt], index=td, columns=["core", "qld", "reserve"])
        w["regime"] = ["공격" if a == "2x" else "방어" for a in tgt]
        res = pf.simulate_target_weights(data, cfg, start, end, w)
        eq = [r for r in res.equity_rows if r["total_krw"] is not None]
        m = bt.compute_equity_metrics(eq)
        a = pf.compute_posttax_a(res, data, cfg, end)
        rows.append({"후보": f"{cand} (L1 엔진)", "세후a %": a.get("cagr_liquidated_pct"), "세후b %": m.get("cagr_pct"),
                     "MDD %": m.get("mdd_pct"), "전환": res.trade_count,
                     "2배 기간 %": round(float((tgt.loc[start:end] == "2x").mean() * 100), 1)})
    return rows


# ── 실행 ─────────────────────────────────────────────────────────────────────


def run_all(ctx: dict, prereg: dict, series_rows, checks, extra) -> None:
    p = prereg["periods"]
    nb_cfg = prereg["neighborhood_values"]
    full, subs, nbs = {}, {}, {}
    for mk in ("N", "U"):
        m = ctx[mk]
        fp = p[f"{mk}_full"]
        start = max(pd.Timestamp(fp["start"]), m["start"])
        print(f"[l1b] {mk} 전 기간 {start.date()} ~ {fp['end']}", flush=True)
        full[mk] = {c: run_daily(m, c, start, fp["end"], pretax=True, **PRIMARY[c]) for c in DAILY}
        subs[mk] = {}
        for s in p[f"{mk}_sub"]:
            s_start = max(pd.Timestamp(s["start"]), m["start"])
            subs[mk][s["name"]] = {c: run_daily(m, c, s_start, s["end"], **PRIMARY[c]) for c in DAILY}
        nbs[mk] = {}
        for c in ("B1", "B2", "B3"):
            lst = []
            for label, params in neighbor_params(c, nb_cfg):
                r = run_daily(m, c, start, fp["end"], **params)
                lst.append((label, r, r["posttax"] - full[mk][c]["posttax"]))
            nbs[mk][c] = lst
    verdict = daily_verdict(full, subs, {mk: {c: [(lb, r["posttax"], d) for lb, r, d in nbs[mk][c]] for c in nbs[mk]} for mk in nbs}, prereg)

    mp = prereg["monthly_candidates"]["period"]
    mstart, mend = pd.Timestamp(mp["start"] + "-01"), pd.Timestamp(mp["end"] + "-01") + pd.offsets.MonthEnd(0)
    print("[l1b] 월별 U", flush=True)
    mfull = {c: run_monthly(ctx, c, mstart, mend, pretax=True) for c in MONTHLY}
    mnb = {}
    for c in ("M1", "M2", "M3"):
        lst = []
        for n in nb_cfg["avg_months"]:
            if n != 10:
                r = run_monthly(ctx, c, mstart, mend, avg_months=n)
                lst.append((f"avg_months={n}", r, r["posttax"] - mfull[c]["posttax"]))
        mnb[c] = lst
    mverdict = monthly_verdict(mfull, {c: [(lb, r["posttax"], d) for lb, r, d in mnb[c]] for c in mnb})

    windows = prereg["reporting"]["crash_windows"]
    crash = crash_rows("U", full["U"], windows["U"]) + crash_rows("N", full["N"], windows["N"])

    print("[l1b] 참고 보고(L1 엔진, QQQ 1999~2021) ...", flush=True)
    try:
        ref = reference_l1_engine()
        ref_err = None
    except Exception as e:  # 참고 보고 실패는 판정과 무관 — 숨기지 않고 보고서에 적는다
        ref, ref_err = [], f"{type(e).__name__}: {e}"
        print(f"[l1b] 참고 보고 실패: {ref_err}", flush=True)

    write_outputs(ctx, prereg, series_rows, checks, extra, full, subs, nbs, verdict, mfull, mnb, mverdict, crash, ref, ref_err)


# ── 출력 ─────────────────────────────────────────────────────────────────────


def _f(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
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


def cand_row(name: str, r: dict, b0: dict | None = None) -> dict:
    return {
        "후보": name, "세후 %": r["posttax"], "세전 %": r["pretax"],
        "기준 대비 세후 %p": (r["posttax"] - b0["posttax"]) if b0 else None,
        "MDD %": r["mdd"], "MDD 고점→저점": f"{r['mdd_peak']} → {r['mdd_trough']}",
        "최장 회복(행)": r["recovery_rows"], "회복 구간": r["recovery_span"] + (" (진행 중)" if r["recovery_open"] else ""),
        "최악의 해": r["worst_year"], "전환": r["switches"], "세금 합계 $": r["tax"], "2배 기간 %": r["lev2_share"],
    }


def make_plots(full: dict, mfull: dict, ctx: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"B0": "#555555", "B1": "#1f77b4", "B2": "#d62728", "B3": "#2ca02c",
              "M0": "#555555", "M1": "#1f77b4", "M2": "#9467bd", "M3": "#ff7f0e"}
    panels = [("U 일별 (1927-07~1998)", full["U"], DAILY), ("N 일별 (1972~1998)", full["N"], DAILY), ("U 월별 (1927-07~1998)", mfull, MONTHLY)]
    paths = []

    fig, axes = plt.subplots(3, 1, figsize=(13, 14))
    for ax, (title, runs, cands) in zip(axes, panels):
        for c in cands:
            v = runs[c]["sim"]["values"]
            ax.plot(v.index, v.values, label=f"{c} 세후 {runs[c]['posttax']:.2f}%", color=colors[c], lw=1)
        ax.set_yscale("log")
        ax.set_title(f"{title} — 자산 곡선(로그, 세후 장부, 시작 $100,000)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "l1b_equity_log.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    for ax, (title, runs, cands) in zip(axes, panels):
        for c in cands:
            v = runs[c]["sim"]["values"]
            ax.plot(v.index, (v / v.cummax() - 1) * 100, label=f"{c} MDD {runs[c]['mdd']:.1f}%", color=colors[c], lw=0.8)
        ax.set_title(f"{title} — 낙폭 %")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "l1b_drawdown.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    paths.append(path)

    band_color = {"2x": "#d62728", "1x": "#bbbbbb", "cash": "#1f77b4", "hidiv": "#9467bd"}
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    for ax, (title, runs, cands), lvl in zip(axes, panels, [ctx["U"]["level"], ctx["N"]["level"], ctx["m_level"]]):
        cs = [c for c in cands if c not in ("B0", "M0")]
        for k, c in enumerate(cs):
            h = runs[c]["sim"]["held"]
            for asset, col in band_color.items():
                mask = (h == asset).to_numpy()
                ax.fill_between(h.index, k, k + 0.9, where=mask, color=col, step="post", linewidth=0)
        ax.set_yticks([k + 0.45 for k in range(len(cs))])
        ax.set_yticklabels(cs)
        ax2 = ax.twinx()
        lv = lvl.loc[runs[cs[0]]["sim"]["held"].index[0]:]
        ax2.plot(lv.index, lv.values, color="black", lw=0.7)
        ax2.set_yscale("log")
        ax.set_title(f"{title} — 국면 띠 (빨강 2배 · 회색 1배 · 파랑 국채 · 보라 고배당) + 지수(로그, 검정)")
    fig.tight_layout()
    path = OUT_DIR / "l1b_regime_bands.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    paths.append(path)
    return paths


def write_outputs(ctx, prereg, series_rows, checks, extra, full, subs, nbs, verdict, mfull, mnb, mverdict, crash, ref, ref_err) -> None:
    import subprocess

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_MD.parent.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    tag = subprocess.run(["git", "rev-parse", "l1b-prereg"], capture_output=True, text=True, cwd=ROOT).stdout.strip()

    # CSV
    pd.DataFrame(series_rows + [{"시리즈": c["항목"], "시작": c["계산 %"], "끝": c["알려진 값 %"]} for c in checks]).to_csv(OUT_DIR / "l1b_data_check.csv", index=False, encoding="utf-8-sig")
    summ = []
    for mk in ("N", "U"):
        for c in DAILY:
            summ.append({"시장": mk, "구간": "전 기간", **cand_row(c, full[mk][c], full[mk]["B0"])})
        for s, runs in subs[mk].items():
            for c in DAILY:
                r = runs[c]
                summ.append({"시장": mk, "구간": s, "후보": c, "세후 %": r["posttax"], "MDD %": r["mdd"], "전환": r["switches"]})
    for c in MONTHLY:
        summ.append({"시장": "U 월별", "구간": "전 기간", **cand_row(c, mfull[c], mfull["M0"])})
    pd.DataFrame(summ).to_csv(OUT_DIR / "l1b_summary.csv", index=False, encoding="utf-8-sig")
    nb_rows = []
    for mk in ("N", "U"):
        for c in ("B1", "B2", "B3"):
            nb_rows.append({"시장": mk, "후보": c, "설정": "주 설정", "세후 %": full[mk][c]["posttax"], "MDD %": full[mk][c]["mdd"], "주 설정 대비 %p": 0.0})
            nb_rows += [{"시장": mk, "후보": c, "설정": lb, "세후 %": r["posttax"], "MDD %": r["mdd"], "주 설정 대비 %p": d} for lb, r, d in nbs[mk][c]]
    for c in ("M1", "M2", "M3"):
        nb_rows.append({"시장": "U 월별", "후보": c, "설정": "주 설정(10개월)", "세후 %": mfull[c]["posttax"], "MDD %": mfull[c]["mdd"], "주 설정 대비 %p": 0.0})
        nb_rows += [{"시장": "U 월별", "후보": c, "설정": lb, "세후 %": r["posttax"], "MDD %": r["mdd"], "주 설정 대비 %p": d} for lb, r, d in mnb[c]]
    pd.DataFrame(nb_rows).to_csv(OUT_DIR / "l1b_neighbors.csv", index=False, encoding="utf-8-sig")
    yearly = {}
    for mk in ("N", "U"):
        for c in DAILY:
            yearly[f"{mk}_{c}"] = full[mk][c]["yearly"]
    for c in MONTHLY:
        yearly[f"U월별_{c}"] = mfull[c]["yearly"]
    ydf = pd.DataFrame(yearly).round(2)
    ydf.to_csv(OUT_DIR / "l1b_yearly.csv", encoding="utf-8-sig")
    pd.DataFrame(crash).to_csv(OUT_DIR / "l1b_crash.csv", index=False, encoding="utf-8-sig")
    eq = {}
    for mk in ("N", "U"):
        for c in DAILY:
            eq[f"{mk}_{c}_value"] = full[mk][c]["sim"]["values"]
            eq[f"{mk}_{c}_held"] = full[mk][c]["sim"]["held"]
    pd.DataFrame(eq).to_csv(OUT_DIR / "l1b_equity_daily.csv", encoding="utf-8-sig")
    pd.DataFrame({**{f"{c}_value": mfull[c]["sim"]["values"] for c in MONTHLY}, **{f"{c}_held": mfull[c]["sim"]["held"] for c in MONTHLY}}).to_csv(OUT_DIR / "l1b_equity_monthly.csv", encoding="utf-8-sig")
    pd.DataFrame(ref).to_csv(OUT_DIR / "l1b_reference_1999_2021.csv", index=False, encoding="utf-8-sig")
    plots = make_plots(full, mfull, ctx)

    # 판정 문장
    passed = [c for c in ("B1", "B2", "B3") if verdict[c]["pass"]]
    vlines = []
    for c in ("B1", "B2", "B3"):
        v = verdict[c]
        mark = lambda k: "✓" if v["ok"][k] else "✗"  # noqa: E731
        vlines.append(
            f"- **{c}: {'합격' if v['pass'] else '불합격'}** — "
            f"1 {mark(1)}(B0 대비 N {v['c1']['N']:+.2f}%p, U {v['c1']['U']:+.2f}%p) · "
            f"2 {mark(2)}(MDD 차이 N {v['c2']['N']:+.2f}%p, U {v['c2']['U']:+.2f}%p) · "
            f"3 {mark(3)}(하위 구간 N {v['c3']['N']}/2, U {v['c3']['U']}/3) · "
            f"4 {mark(4)}(주변값 차이 N {[round(d, 2) for d in v['c4']['N']]}, U {[round(d, 2) for d in v['c4']['U']]})"
        )
    if passed:
        worst_mdd = {c: min(full["N"][c]["mdd"], full["U"][c]["mdd"]) for c in passed}
        final = max(passed, key=lambda c: worst_mdd[c])
        next_step = f"**{final}** (N·U 중 더 깊은 MDD {worst_mdd[final]:.2f}%) → paper 기록 + 봉인 구간(2022~) 1회 개봉"
    else:
        next_step = "**없음 — 일별 후보 전부 불합격. 그대로 기록하고 같은 기간(N 1972~1998, U 1927~1998)을 다시 판정에 쓰지 않는다.**"
    mpass = [c for c in ("M1", "M2", "M3") if mverdict[c]["pass"]]
    mlines = []
    for c in ("M1", "M2", "M3"):
        v = mverdict[c]
        mark = lambda k: "✓" if v["ok"][k] else "✗"  # noqa: E731
        mlines.append(f"- **{c}({ASSET_KO[MONTHLY_DEFENSE[c]]} 방어): {'충족' if v['pass'] else '미충족'}** — "
                      f"1 {mark(1)}(M0 대비 {v['c1']:+.2f}%p) · 2 {mark(2)}(MDD 차이 {v['c2']:+.2f}%p) · "
                      f"4 {mark(4)}(9·11개월 차이 {[round(d, 2) for d in v['c4']]})")
    if mpass:
        best = max(mpass, key=lambda c: mfull[c]["posttax"])
        mdecision = f"**방어 자산 후보: {ASSET_KO[MONTHLY_DEFENSE[best]]}** ({best}, 세후 {mfull[best]['posttax']:.2f}%)"
    else:
        mdecision = "**월별 10개월 평균 2배 전환 불합격** (M1~M3 모두 판정 1·2·4 중 하나 이상 미충족)"

    def daily_table(mk):
        return md_table([cand_row(c, full[mk][c], full[mk]["B0"]) for c in DAILY])

    def sub_table(mk):
        rows = []
        for c in DAILY:
            row = {"후보": c}
            for s, runs in subs[mk].items():
                row[f"{s} 세후 %"] = runs[c]["posttax"]
                row[f"{s} MDD %"] = runs[c]["mdd"]
            if c != "B0":
                row["B0보다 높은 구간"] = f"{sum(runs_[c]['posttax'] > runs_['B0']['posttax'] for runs_ in subs[mk].values())}/{len(subs[mk])}"
            else:
                row["B0보다 높은 구간"] = "-"
            rows.append(row)
        return md_table(rows)

    nb_md = md_table([{k: v for k, v in r.items()} for r in nb_rows])
    crash_md = md_table(crash)
    ydf_md = ydf.copy()
    ydf_md.insert(0, "연도", ydf_md.index)
    yearly_md = md_table(ydf_md.to_dict("records"))

    md = f"""# L1b 새 기간 국면 검증 — 결과 (L1b 시도 1)

- 사전 등록: `docs/l1b_plan.md`, `configs/l1b_preregistration.yaml` (tag `l1b-prereg` = `{tag}`), 시작 시 sha256 확인 통과
  - `docs/l1b_plan.md` `{PREREG_FILES['docs/l1b_plan.md']}`
  - `configs/l1b_preregistration.yaml` `{PREREG_FILES['configs/l1b_preregistration.yaml']}`
- 실행 코드 커밋: `{head}` · 봉인 1998-12-31(1999-01-01 이후 행 검사 통과) · 같은 기간 1999~2021은 판정에 쓰지 않음
- **가상 상품 주의**: 2배는 실제 상품이 없던 시기의 가상 합성 상품(일간 2×지수 − RF − 0.95%/252, 매일 재조정)이다.
- **체결 규칙**: 일별 후보는 신호 = 그날 종가, 체결 = 다음 거래일 **종가** (L1은 다음 거래일 **시가**였다). 월별은 신호 월말 종가에 바로 체결.

## 판정 — 일별 후보 (N·U 모두)

{chr(10).join(vlines)}

최종 후보: {next_step}

## 판정 — 월별 방어 자산 (U 1927-07~1998-12)

{chr(10).join(mlines)}

결론: {mdecision}

## 데이터 검사표

{md_table(series_rows)}
{md_table(checks)}
{md_table(extra)}

## 일별 후보 — U (미국 전체 시장 총수익, {full['U']['B0']['sim']['values'].index[0].date()} ~ 1998-12-31)

{daily_table('U')}
하위 구간(각 구간 시작에 $100,000로 새로 시작, 끝에 청산·세금):

{sub_table('U')}
## 일별 후보 — N (나스닥 종합 가격, 배당 미포함, {full['N']['B0']['sim']['values'].index[0].date()} ~ 1998-12-31)

{daily_table('N')}
하위 구간:

{sub_table('N')}
## 주변값 (전 기간 세후, 한 번에 하나만 바꿈, 안정 = 모두 ±1%p 이내)

{nb_md}
## 월별 후보 — U (월말 가치 기준 MDD·회복(월))

{md_table([cand_row(c, mfull[c], mfull['M0']) for c in MONTHLY])}
## 폭락 구간 대응 (전 기간 시뮬레이션, 창 손실 = 창 시작 전날 종가 대비 창 끝, 전환일 = 체결일)

{crash_md}
## 참고 보고 — QQQ 1999-03-10 ~ 2021-12-31, L1 엔진·세금 그대로 (판정에 쓰지 않음)

L1 엔진 규칙(다음 날 **시가** 체결, 원화 4,000만 원, 양도세 250만 원 공제·5월 납부, 배당 원천 15%, 1배 = QQQM, 2배 = QLD(2006-06 전 합성), 국채 = DTB3). 신호 지수 = ^NDX 접합 QQQ 종가.
{md_table(ref) if ref else f"실행 실패: {ref_err}"}
## 연도별 수익 (세후 장부 평가액, %)

{yearly_md}
## 결과를 본 뒤 바꾼 점

- 없음 (사전 등록과 구현 해석 그대로 1회 실행).

## 한계

- 2배는 가상 합성 상품(실제 상품 없음). 차입 비용 = RF, 보수 0.95%/252를 행마다(1952년 이전 토요일 행 포함 → 그 시기 보수가 연 0.95%보다 조금 큼).
- 배당 원천세 미반영. U·고배당의 배당은 총수익에 포함된 채 과세하지 않음. 국채 이자는 실현 소득으로 과세(보수적).
- N(나스닥 종합)은 가격 지수라 배당 미포함 — 모든 N 후보에 똑같이 불리.
- 일별 체결은 다음 거래일 종가(L1의 다음 날 시가와 다름). 1987-10-19 같은 하루 급락은 신호 다음 날 종가까지 피할 수 없다.
- U는 CRSP 전체 시장(S&P 500 대용)이고 신호도 총수익 지수로 판단(실제 투자자가 보는 가격 지수와 조금 다름).
- 세금은 달러 기준 근사(환율·공제·손실 이월 없음, 연말 납부).
- 폭락 표본이 적다(U 6개, N 4개 창). 합격해도 "방향이 맞다" 수준이다.

## 그래프

{chr(10).join(f'- `{p.relative_to(ROOT).as_posix()}`' for p in plots)}
"""
    RESULT_MD.write_text(md, encoding="utf-8")
    print(f"[l1b] 결과: {RESULT_MD.relative_to(ROOT)}", flush=True)
    print("\n".join(vlines))
    print("최종 후보:", next_step)
    print("\n".join(mlines))
    print("월별 결론:", mdecision)


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n## {title}")
    if not rows:
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
    prereg = verify_preregistration()
    ctx = load_markets(prereg)
    series_rows, checks, extra = data_check(ctx, prereg)
    print_table(series_rows, "데이터 검사표 — 시리즈")
    print_table(checks, "데이터 검사표 — 알려진 값 대조")
    print_table(extra, "데이터 검사표 — 정렬·워밍업")
    if args.check_data:
        return
    run_all(ctx, prereg, series_rows, checks, extra)


if __name__ == "__main__":
    main()
