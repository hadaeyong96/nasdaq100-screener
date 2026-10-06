"""L1 레버리지 국면 전환 과거 검증 (사전 등록: docs/l1_plan.md, configs/l1_preregistration.yaml, tag l1-prereg).

일회성 리포트 스크립트(CLAUDE.md scripts/). 국면 신호·목표 비중은 core/regime.py(순수 함수),
시뮬레이션·세금은 engine/portfolio.py(P6 엔진 재사용 + simulate_target_weights)를 쓴다.
여기서는 데이터 준비·봉인·후보/주변값/구간 조합·판정·보고만 한다.

시작할 때 사전 등록 두 파일의 sha256(줄바꿈 LF로 맞춘 내용)을 확인하고 다르면 멈춘다.
2022-01-01 이후 데이터는 로딩 단계에서 자르고, 한 행이라도 남으면 멈춘다(core.regime.enforce_seal).

실행: python -u -m scripts.l1_experiments
"""

from __future__ import annotations

import hashlib
import pickle
import subprocess
import sys
from datetime import date
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

from core import regime as rg  # noqa: E402
from data import fx as fxmod  # noqa: E402
from engine import backtest as bt  # noqa: E402
from engine import portfolio as pf  # noqa: E402

PREREG_FILES = {
    "docs/l1_plan.md": "95bf23d3fda942095f0f4205838420a58204f36718ce545e85dfdd322e0150f7",
    "configs/l1_preregistration.yaml": "1029f7a110fe25dee2b478893bee7e8e10a842de593d95569230d2d7a1dafb1d",
}
CHECKPOINT_DIR = bt.OUTPUT_ROOT / "l1_checkpoint"
OUT_DIR = ROOT / "outputs"
RESULT_MD = ROOT / "docs" / "results" / "l1.md"
CANDIDATES = ["L0", "L1", "L2", "L3", "L4"]
# 결과를 보고 떠오른 아이디어 — 판정에 쓰지 않고, 이 기간(1999~2021)으로 다시 시험하지 않는다.
NEW_IDEAS = [
    "- 방어 국면에서 국채 대신 QQQM 100%로만 내리기(레버리지만 끄기) — 2000~2002 장기 하락에서 QLD 손실을 줄이는지",
    "- 200일선 이탈을 n일 연속 확인한 뒤 전환(휩소 감소) — L2는 154번 전환, 폭락 창마다 하루 만에 되돌아온 첫 탈출이 많았다",
    "- 계좌 낙폭 정지(예: 고점 대비 −30%면 레버리지 해제) — 모든 후보의 MDD가 2000-03 → 2002/2003 한 구간에서 나왔다",
    "- BAA 스프레드 1년 평균 위 또는 장단기 금리차 역전을 방어 조건에 더하기 — 참고 분할에서 그 국면의 다음 6개월 하위 10%가 크게 나빴다",
    "- 변동성 목표 레버리지(최근 변동성이 높을수록 QLD 비중 축소)",
    "- 실업률 당시 발표치(ALFRED vintage)로 L3 규칙 재확인",
]


# ── 사전 등록 확인 ───────────────────────────────────────────────────────────


def normalized_sha256(path: Path) -> str:
    """줄바꿈을 LF로 맞춘 내용의 sha256 (Windows 체크아웃의 CRLF 변환에 흔들리지 않게)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_preregistration() -> dict:
    for rel, expected in PREREG_FILES.items():
        got = normalized_sha256(ROOT / rel)
        if got != expected:
            raise SystemExit(f"[l1] 사전 등록 파일이 바뀌었습니다: {rel} (기대 {expected[:12]}…, 실제 {got[:12]}…) — 멈춤")
    with open(ROOT / "configs" / "l1_preregistration.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 데이터 ───────────────────────────────────────────────────────────────────


def _ckpt(name: str, fn):
    path = CHECKPOINT_DIR / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            print(f"[checkpoint] {name} <- 불러옴", flush=True)
            return pickle.load(f)
    print(f"[checkpoint] {name} 계산 중 ...", flush=True)
    obj = fn()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return obj


def fred_series(series_id: str, start: date, end: date) -> pd.Series:
    """FRED 시계열을 end(봉인일)까지만 받아 봉인 검사 후 Series로 돌려준다."""
    raw = fxmod.fetch_fred_series_range(series_id, start, end)
    raw = rg.truncate_to_seal(raw, rg.LAST_ALLOWED_DATE)
    rg.enforce_seal(raw, rg.LAST_ALLOWED_DATE, series_id)
    s = pd.Series(raw, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def load_all(cfg: dict, prereg: dict) -> dict:
    p = prereg["periods"]["full"]
    start, end = date.fromisoformat(p["start"]), date.fromisoformat(p["end"])
    if pd.Timestamp(end) > rg.LAST_ALLOWED_DATE:
        raise rg.SealError("기간 끝이 봉인일 뒤입니다")
    data = _ckpt("data", lambda: pf.prepare_data(cfg, start, end))
    data = pf.seal_portfolio_data(data, rg.LAST_ALLOWED_DATE)
    unrate = _ckpt("unrate", lambda: fred_series("UNRATE", date(1996, 1, 1), end))
    t10y2y = _ckpt("t10y2y", lambda: fred_series("T10Y2Y", date(1997, 1, 1), end))
    baa = _ckpt("baa10y", lambda: fred_series("BAA10Y", date(1997, 1, 1), end))
    for name, s in (("unrate", unrate), ("t10y2y", t10y2y), ("baa10y", baa)):
        rg.enforce_seal(s, rg.LAST_ALLOWED_DATE, name)
    return {"data": data, "unrate": unrate, "t10y2y": t10y2y, "baa10y": baa, "start": start, "end": end}


# ── 신호 ─────────────────────────────────────────────────────────────────────


def weights_for(ctx: dict, candidate: str, sma_days: int = 200, unrate_months: int = 12) -> pd.DataFrame:
    data = ctx["data"]
    td = data.qqq_df.index
    above = rg.above_sma(data.sma_source_df["close"], sma_days).reindex(td)
    unrate_up = rg.monthly_above_avg(ctx["unrate"], td, unrate_months) if candidate in ("L3", "L4") else None
    return rg.target_weights(candidate, above, unrate_up)


# ── 시나리오 ─────────────────────────────────────────────────────────────────


def run_scenario(ctx: dict, cfg: dict, candidate: str, start: date, end: date, **params) -> dict:
    data = ctx["data"]
    if candidate == "L0":  # 기준선은 P6-1 P0 엔진 그대로(재현 확인과 같은 경로)
        result = pf.simulate_portfolio(data, cfg, start, end, "P0")
        weights = None
    else:
        weights = weights_for(ctx, candidate, **params)
        result = pf.simulate_target_weights(data, cfg, start, end, weights)
    rows = [r for r in result.equity_rows if r["total_krw"] is not None]
    m = bt.compute_equity_metrics(rows)
    a = pf.compute_posttax_a(result, data, cfg, end)
    regimes = pd.Series([r.get("regime", "공격") for r in rows])
    years = (pd.Timestamp(end) - pd.Timestamp(rows[0]["date"])).days / 365.25 if rows else 0
    exact_a = ((a["liquidated_value_krw"] / cfg["backtest"]["total_krw"]) ** (1 / years) - 1) * 100 if years and a.get("liquidated_value_krw") else None
    v = pd.Series([r["total_krw"] for r in rows], index=[r["date"] for r in rows])
    ddser = v / v.cummax() - 1
    trough = ddser.idxmin() if len(v) else None
    return {
        "posttax_a_exact": exact_a,
        "mdd_peak": v.loc[:trough].idxmax() if trough else None, "mdd_trough": trough,
        "candidate": candidate, "start": start, "end": end, "params": params, "rows": rows, "weights": weights,
        "posttax_a_pct": a.get("cagr_liquidated_pct"), "posttax_b_pct": m.get("cagr_pct"), "mdd_pct": m.get("mdd_pct"),
        "recovery": pf.compute_longest_recovery_days(rows), "worst_year": pf.compute_worst_year_month(rows)["worst_year"],
        "yearly_pct": bt.compute_yearly_returns_from_equity(rows), "switches": result.trade_count,
        "total_tax_krw": round(sum(t["tax_krw"] for t in result.broker.tax_log) + (a.get("tax_paid_krw") or 0)),
        "attack_share_pct": round(float((regimes == "공격").mean() * 100), 1) if len(regimes) else None,
    }


# ── 폭락 구간 탈출·복귀 ───────────────────────────────────────────────────────


def crash_exits(weights: pd.DataFrame, qqq_close: pd.Series, windows: dict, lookback: int) -> dict:
    """폭락 창마다 탈출·복귀 (사전 등록 구현 해석 11번 그대로).

    탈출 = 창 시작 lookback거래일 전 ~ 창 끝 사이에서 처음 공격→방어로 바뀐 신호일, 복귀 = 그 뒤 처음 방어→공격
    신호일. 낙폭은 두 기준을 함께 적는다: QQQ 직전 1년(252거래일) 최고 종가 대비, 사상 최고 종가 대비.
    보조: 그 구간 시작 때 이미 방어였는지, 구간 안 방어 전환 횟수, 창 안 방어(보유) 비율.
    """
    ath = qqq_close.cummax()
    high_1y = qqq_close.rolling(252, min_periods=1).max()
    dd_1y = (qqq_close / high_1y - 1) * 100
    dd_ath = (qqq_close / ath - 1) * 100
    reg = weights["regime"]
    idx = reg.index
    prev = reg.shift(1)
    exits = reg[(reg == "방어") & (prev == "공격")].index
    returns = reg[(reg == "공격") & (prev == "방어")].index
    out = {}
    for label, (ws, we) in windows.items():
        ws, we = pd.Timestamp(ws), pd.Timestamp(we)
        span_start = idx[max(idx.searchsorted(ws) - lookback, 0)]
        in_span_exits = exits[(exits >= span_start) & (exits <= we)]
        exit_day = in_span_exits[0] if len(in_span_exits) else None
        later = returns[returns > exit_day] if exit_day is not None else []
        ret_day = later[0] if len(later) else None
        win = reg.loc[ws:we]
        out[label] = {
            "구간 시작 때 이미 방어": bool(reg.loc[span_start] == "방어"),
            "탈출 신호일": exit_day.date().isoformat() if exit_day is not None else None,
            "탈출 때 QQQ 1년고점 대비 %": round(float(dd_1y.loc[exit_day]), 1) if exit_day is not None else None,
            "탈출 때 QQQ 사상최고 대비 %": round(float(dd_ath.loc[exit_day]), 1) if exit_day is not None else None,
            "복귀 신호일": ret_day.date().isoformat() if ret_day is not None else None,
            "복귀 때 QQQ 1년고점 대비 %": round(float(dd_1y.loc[ret_day]), 1) if ret_day is not None else None,
            "구간 안 방어 전환 횟수": int(len(in_span_exits)),
            "창 안 방어 비율 %": round(float((win == "방어").mean() * 100), 1) if len(win) else None,
            "창 QQQ 1년고점 대비 최저 %": round(float(dd_1y.loc[ws:we].min()), 1),
        }
    return out


# ── 국면 분석 (설명용) ─────────────────────────────────────────────────────────


def _max_dd_within_runs(close: pd.Series, mask: pd.Series) -> float | None:
    worst = 0.0
    run_id = (mask != mask.shift()).cumsum()
    for _, seg in close[mask].groupby(run_id[mask]):
        if len(seg) > 1:
            worst = min(worst, float((seg / seg.cummax() - 1).min()))
    return round(worst * 100, 1) if mask.any() else None


def regime_analysis(ctx: dict) -> dict:
    """4국면(200일선 위/아래 × 실업률 상승/하락) + 참고 분할(T10Y2Y 역전, BAA10Y 1년 평균 위/아래)."""
    data = ctx["data"]
    td = data.qqq_df.index[(data.qqq_df.index >= pd.Timestamp(ctx["start"])) & (data.qqq_df.index <= pd.Timestamp(ctx["end"]))]
    qqq = data.qqq_df["close"].reindex(td)
    qld = data.qld_df["close"].reindex(td)
    above = rg.above_sma(data.sma_source_df["close"], 200).reindex(td)
    up = rg.monthly_above_avg(ctx["unrate"], td, 12)
    t10 = rg.lagged_daily(ctx["t10y2y"], td, 1)
    baa = rg.lagged_daily(ctx["baa10y"], td, 1)
    baa_avg = baa.rolling(252, min_periods=252).mean()
    splits = {
        "200일선 위 · 실업률 하락": (above == True) & ~up,  # noqa: E712
        "200일선 위 · 실업률 상승": (above == True) & up,  # noqa: E712
        "200일선 아래 · 실업률 하락": (above == False) & ~up,  # noqa: E712
        "200일선 아래 · 실업률 상승": (above == False) & up,  # noqa: E712
        "참고: 장단기 금리차 역전": t10 < 0,
        "참고: 장단기 금리차 정상": t10 >= 0,
        "참고: BAA 스프레드 1년 평균 위": baa > baa_avg,
        "참고: BAA 스프레드 1년 평균 아래": baa <= baa_avg,
    }
    horizons = {"1개월": 21, "3개월": 63, "6개월": 126}
    rows = []
    for name, mask in splits.items():
        mask = mask.fillna(False).astype(bool)
        r = {"국면": name, "기간 비율 %": round(float(mask.mean() * 100), 1), "일수": int(mask.sum())}
        for asset, close in (("QQQ", qqq), ("QLD", qld)):
            daily = close.pct_change()
            r[f"{asset} 연변동성 %"] = round(float(daily[mask].std() * np.sqrt(252) * 100), 1) if mask.sum() > 2 else None
            r[f"{asset} 국면 안 최대낙폭 %"] = _max_dd_within_runs(close, mask)
            for hname, n in horizons.items():
                fwd = (close.shift(-n) / close - 1) * 100  # 설명용 사후 분석(신호에 쓰지 않음), 2021-12-31 뒤는 NaN
                v = fwd[mask].dropna()
                r[f"{asset} {hname} 평균"] = round(float(v.mean()), 1) if len(v) else None
                r[f"{asset} {hname} 중앙값"] = round(float(v.median()), 1) if len(v) else None
                r[f"{asset} {hname} 하위10%"] = round(float(v.quantile(0.1)), 1) if len(v) else None
        rows.append(r)
    return {"table": pd.DataFrame(rows)}


# ── 판정 ─────────────────────────────────────────────────────────────────────


def judge(main: dict, subs: dict, neighbors: dict, prereg: dict) -> dict:
    thr = {c["id"]: c for c in prereg["criteria"]}
    out = {}
    l0 = main["L0"]
    for cand in CANDIDATES[1:]:
        s = main[cand]
        excess_exact = (s["posttax_a_exact"] or 0) - (l0["posttax_a_exact"] or 0)  # 판정은 반올림 전 값으로
        excess = round(excess_exact, 2)
        c1 = excess_exact >= thr[1]["min"]
        c2 = (s["mdd_pct"] or -100) >= thr[2]["min"]
        hits = [k for k in subs if (subs[k][cand]["posttax_a_pct"] or -99) > (subs[k]["L0"]["posttax_a_pct"] or -99)]
        c3 = len(hits) >= thr[3]["min_count"]
        nb = neighbors.get(cand, [])
        nb_diffs = [round((n["posttax_a_pct"] or 0) - (s["posttax_a_pct"] or 0), 2) for n in nb]
        c4 = all(abs(d) <= prereg["neighborhood_values"]["stable_if_all_neighbors_within_pp"] for d in nb_diffs)
        reasons = []
        if not c1:
            reasons.append(f"1 미충족: 세후(a) L0 대비 {excess:+.2f}%p (< +2.00%p)")
        if not c2:
            reasons.append(f"2 미충족: MDD {s['mdd_pct']}% (< −50%)")
        if not c3:
            reasons.append(f"3 미충족: 하위 구간 {len(hits)}/3")
        if not c4:
            reasons.append(f"4 미충족: 주변값 차이 {nb_diffs}")
        out[cand] = {"excess_pp": excess, "excess_exact_pp": round(excess_exact, 4), "c1": c1, "c2": c2, "c3": c3, "c3_hits": hits, "c4": c4,
                     "neighbor_diffs": nb_diffs, "pass": c1 and c2 and c3 and c4, "reasons": reasons}
    passed = [c for c in out if out[c]["pass"]]
    selected = min(passed, key=lambda c: abs(main[c]["mdd_pct"])) if passed else None
    return {"per_candidate": out, "selected": selected}


# ── 그래프 ───────────────────────────────────────────────────────────────────


def plot_all(main: dict) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    paths = []
    colors = {"L0": "#444444", "L1": "#c0392b", "L2": "#2e86c1", "L3": "#27ae60", "L4": "#8e44ad"}

    fig, ax = plt.subplots(figsize=(12, 6))
    for c in CANDIDATES:
        df = pd.DataFrame(main[c]["rows"])
        ax.plot(pd.to_datetime(df["date"]), df["total_krw"] / 1e6, label=c, color=colors[c], lw=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("평가액 (백만 원, 로그)")
    ax.set_title("L1 후보별 자산 곡선 (1999-03 ~ 2021-12, 세금·비용 반영 평가액)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    p = OUT_DIR / "l1_equity_log.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(12, 5))
    for c in CANDIDATES:
        df = pd.DataFrame(main[c]["rows"])
        v = df["total_krw"]
        ax.plot(pd.to_datetime(df["date"]), (v / v.cummax() - 1) * 100, label=c, color=colors[c], lw=1.0)
    ax.axhline(-50, color="black", ls="--", lw=0.8)
    ax.set_ylabel("고점 대비 낙폭 (%)")
    ax.set_title("낙폭 곡선 (점선 = 판정 기준 −50%)")
    ax.legend()
    ax.grid(alpha=0.3)
    p = OUT_DIR / "l1_drawdown.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True)
    for ax, c in zip(axes, ("L2", "L3", "L4")):
        df = pd.DataFrame(main[c]["rows"])
        dates = pd.to_datetime(df["date"])
        is_def = (df["regime"] == "방어").astype(int)
        ax.fill_between(dates, 0, 1, where=is_def == 0, color="#27ae60", alpha=0.5, step="post", label="공격")
        ax.fill_between(dates, 0, 1, where=is_def == 1, color="#c0392b", alpha=0.5, step="post", label="방어(국채)")
        ax.set_yticks([])
        ax.set_ylabel(c, rotation=0, labelpad=15)
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    axes[0].set_title("국면 띠 (보유 기준: 초록 = 공격, 빨강 = 방어)")
    p = OUT_DIR / "l1_regime_bands.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths


# ── 메인 ─────────────────────────────────────────────────────────────────────


def main() -> None:
    prereg = verify_preregistration()
    print("[l1] 사전 등록 해시 확인 완료", flush=True)
    cfg = bt.load_config()
    ctx = load_all(cfg, prereg)
    data = ctx["data"]

    # 재현 확인: P6-1 P0(전 기간)
    ref = prereg["reproduction_check"]["reference"]
    tol = prereg["reproduction_check"]["tolerance_pp"]
    repro = run_scenario(ctx, cfg, "L0", ctx["start"], ctx["end"])
    if abs(repro["posttax_a_pct"] - ref["posttax_a_pct"]) > tol or abs(repro["mdd_pct"] - ref["mdd_pct"]) > tol:
        raise SystemExit(f"[l1] P6-1 재현 실패: 세후(a) {repro['posttax_a_pct']} vs {ref['posttax_a_pct']}, MDD {repro['mdd_pct']} vs {ref['mdd_pct']} — 멈춤")
    print(f"[l1] P6-1 P0 재현: 세후(a) {repro['posttax_a_pct']}%, MDD {repro['mdd_pct']}%", flush=True)
    # 교차 확인: 같은 L0를 새 목표 비중 엔진으로
    w0 = weights_for(ctx, "L0")
    r0 = pf.simulate_target_weights(data, cfg, ctx["start"], ctx["end"], w0)
    r0_rows = [r for r in r0.equity_rows if r["total_krw"] is not None]
    l0_new = {"posttax_a_pct": pf.compute_posttax_a(r0, data, cfg, ctx["end"])["cagr_liquidated_pct"],
              "mdd_pct": bt.compute_equity_metrics(r0_rows)["mdd_pct"]}

    periods = {k: (date.fromisoformat(v["start"]), date.fromisoformat(v["end"]))
               for k, v in prereg["periods"].items() if isinstance(v, dict)}
    main_runs = {c: run_scenario(ctx, cfg, c, *periods["full"]) for c in CANDIDATES}
    subs = {k: {c: run_scenario(ctx, cfg, c, *periods[k]) for c in CANDIDATES} for k in ("sub_1999_2007", "sub_2008_2015", "sub_2016_2021")}

    nv = prereg["neighborhood_values"]
    neighbors: dict[str, list] = {}
    for cand, dims in nv["applies_to"].items():
        lst = []
        for dim in dims:
            for val in nv[dim]:
                if val == prereg["candidates"][cand]["params"][dim]:
                    continue
                params = {"sma_days": 200, "unrate_months": 12}
                params["sma_days" if dim == "sma_days" else "unrate_months"] = val
                r = run_scenario(ctx, cfg, cand, *periods["full"], **params)
                r["neighbor"] = f"{dim}={val}"
                lst.append(r)
        neighbors[cand] = lst

    verdict = judge(main_runs, subs, neighbors, prereg)

    windows = prereg["reporting"]["crash_windows"]
    lookback = prereg["reporting"]["crash_exit_lookback_trading_days"]
    qqq_close = data.qqq_df["close"]
    crash = {c: crash_exits(main_runs[c]["weights"], qqq_close, windows, lookback) for c in ("L2", "L3", "L4")}
    regimes = regime_analysis(ctx)

    write_outputs(cfg, prereg, ctx, repro, l0_new, main_runs, subs, neighbors, verdict, crash, regimes)


def _md_table(df: pd.DataFrame, index: bool = False) -> str:
    """DataFrame -> 마크다운 표 (tabulate 의존 없이)."""
    d = df.reset_index() if index else df

    def cell(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        if isinstance(v, float):
            return f"{v:,.2f}" if abs(v) < 1e6 else f"{v:,.0f}"
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            return f"{v:,}" if abs(int(v)) >= 10000 else str(v)
        return str(v)

    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _pct(v) -> str:
    return f"{v:+.2f}%" if v is not None else "-"


def write_outputs(cfg, prereg, ctx, repro, l0_new, main_runs, subs, neighbors, verdict, crash, regimes) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    RESULT_MD.parent.mkdir(parents=True, exist_ok=True)
    data = ctx["data"]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        tag = subprocess.run(["git", "rev-parse", "l1-prereg"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        head = tag = None

    summary = []
    for c in CANDIDATES:
        s = main_runs[c]
        rec = s["recovery"]
        summary.append({
            "후보": c, "세후a %": s["posttax_a_pct"], "세후b %": s["posttax_b_pct"], "MDD %": s["mdd_pct"],
            "L0 대비 a %p": round((s["posttax_a_pct"] or 0) - (main_runs["L0"]["posttax_a_pct"] or 0), 2),
            "MDD 고점→저점": f"{s['mdd_peak']} → {s['mdd_trough']}",
            "최장 회복(거래일)": rec.get("longest_recovery_trading_days"), "회복 진행중": rec.get("open_ended"),
            "최악의 해": f"{s['worst_year'][0]} {s['worst_year'][1]:+.1f}%" if s["worst_year"] else None,
            "전환 횟수": s["switches"], "세금 합계(원)": s["total_tax_krw"], "공격 기간 %": s["attack_share_pct"],
            **{f"{k} 세후a %": subs[k][c]["posttax_a_pct"] for k in subs},
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / "l1_summary.csv", index=False, encoding="utf-8-sig")

    nb_rows = []
    for cand, lst in neighbors.items():
        nb_rows.append({"후보": cand, "설정": "주 설정", "세후a %": main_runs[cand]["posttax_a_pct"], "MDD %": main_runs[cand]["mdd_pct"], "주 설정 대비 %p": 0.0})
        for r in lst:
            nb_rows.append({"후보": cand, "설정": r["neighbor"], "세후a %": r["posttax_a_pct"], "MDD %": r["mdd_pct"],
                            "주 설정 대비 %p": round((r["posttax_a_pct"] or 0) - (main_runs[cand]["posttax_a_pct"] or 0), 2)})
    nb_df = pd.DataFrame(nb_rows)
    nb_df.to_csv(OUT_DIR / "l1_neighbors.csv", index=False, encoding="utf-8-sig")

    yearly = pd.DataFrame({c: main_runs[c]["yearly_pct"] for c in CANDIDATES})
    yearly.index.name = "연도"
    yearly.to_csv(OUT_DIR / "l1_yearly.csv", encoding="utf-8-sig")

    crash_rows = [{"후보": c, "구간": k, **v} for c, d in crash.items() for k, v in d.items()]
    crash_df = pd.DataFrame(crash_rows)
    crash_df.to_csv(OUT_DIR / "l1_crash_exits.csv", index=False, encoding="utf-8-sig")
    regimes["table"].to_csv(OUT_DIR / "l1_regimes.csv", index=False, encoding="utf-8-sig")
    eq = pd.DataFrame({c: pd.Series({r["date"]: r["total_krw"] for r in main_runs[c]["rows"]}) for c in CANDIDATES})
    eq.index.name = "date"
    eq.to_csv(OUT_DIR / "l1_equity.csv", encoding="utf-8-sig")
    pngs = plot_all(main_runs)

    v = verdict["per_candidate"]
    md = [
        "# L1 레버리지 국면 전환 — 과거 검증 결과 (시도 1)", "",
        f"- 사전 등록: `docs/l1_plan.md`, `configs/l1_preregistration.yaml` (tag `l1-prereg` = `{tag}`), 해시 확인 통과",
        f"- 실행 코드 커밋: `{head}` · 기간 {ctx['start']} ~ {ctx['end']} (2022-01-01 이후 봉인, 로딩 단계 검사 통과)",
        f"- P6-1 재현: P0 세후(a) {repro['posttax_a_pct']}% · MDD {repro['mdd_pct']}% (기준 8.88% · −80.91%) → **일치**",
        f"- 교차 확인: 같은 L0를 새 목표 비중 엔진으로 돌리면 세후(a) {l0_new['posttax_a_pct']}% · MDD {l0_new['mdd_pct']}% "
        "(첫날 수수료 처리 차이 수준). 판정의 L0는 P6 엔진 값을 쓴다.",
        "", "## 판정", "",
    ]
    for c in CANDIDATES[1:]:
        r = v[c]
        conds = (f"1 {'✓' if r['c1'] else '✗'}(L0 대비 {r['excess_exact_pp']:+.4f}%p) · 2 {'✓' if r['c2'] else '✗'}(MDD {main_runs[c]['mdd_pct']}%) · "
                 f"3 {'✓' if r['c3'] else '✗'}(하위 구간 {len(r['c3_hits'])}/3) · 4 {'✓' if r['c4'] else '✗'}(주변값 차이 {[float(x) for x in r['neighbor_diffs']] or '해당 없음'})")
        md.append(f"- **{c}: {'합격' if r['pass'] else '불합격'}** — {conds}")
    md += ["", f"다음 단계 후보: **{verdict['selected'] or '없음 (전 후보 불합격 — 같은 기간 재실행 금지)'}**", ""]

    md += ["## 후보별 결과 (전 기간 1999-03 ~ 2021-12)", "", _md_table(summary_df), ""]
    md += ["## 주변값 (전 기간 세후 (a), 한 번에 하나만 바꿈, 안정 = 모두 ±1%p 이내)", "", _md_table(nb_df), ""]
    md += ["## 폭락 구간 탈출·복귀 (신호일 기준, 체결은 다음 거래일 시가 · 탈출 = 창 시작 60거래일 전부터 처음 방어 전환)", "",
           _md_table(crash_df), ""]
    md += ["## 연도별 수익 (평가액 기준 %, 미청산)", "", _md_table(yearly.round(1), index=True), ""]
    md += ["## 국면 분석 (설명용 — 판정에 쓰지 않음)", "",
           "4국면 = QQQ 200일선 위/아래 × 실업률이 12개월 평균 위(상승)/아래(하락). 실업률은 M월 값을 M+2월 첫 거래일부터, "
           "T10Y2Y·BAA10Y는 하루 늦게 사용. 다음 1·3·6개월 수익은 사후 통계(신호 아님).", "",
           _md_table(regimes["table"]), ""]
    qc = data.qld_synthesis_check
    md += ["## 데이터 점검", "",
           f"- 합성 QLD vs 실제 QLD(2006-06 ~ 2021-12, {qc.get('overlap_days')}거래일): 연 수익 차이 {qc.get('annual_return_diff_pp')}%p "
           f"(합성이 {'높음' if qc.get('synthetic_better_than_real') else '낮음'}), 일별 추적 오차(표준편차) {qc.get('tracking_error_daily_std_pct')}%",
           f"- 환율 보완: {data.fx_fallback_stats}",
           "", "## 한계", "",
           "- 실업률은 현재 수정치(revised)를 쓴다. 당시 처음 발표된 값과 다를 수 있다(발표 지연은 M+2월 첫 거래일로 보수 처리).",
           "- 2006-06 전 QLD는 합성(시가 = 종가)이라 그 구간의 다음 날 시가 체결은 종가 체결과 같다.",
           "- 단기 국채(DTB3)는 이자소득세를 반영하지 않았다(P6와 같은 가정) — 방어 비중이 큰 후보의 세후 값이 그만큼 높게 나온다.",
           "", "## 새 판 후보 (이번 판정에 쓰지 않음, 같은 기간 재실행 금지)", "",
           "- 이 목록은 결과를 보고 떠오른 아이디어일 뿐이며, 새 기간(2022 이후 봉인 해제 전 별도 사전 등록)으로만 시험할 수 있다.",
           "", *NEW_IDEAS, "",
           "## 파일", "", *[f"- `{p.relative_to(ROOT).as_posix()}`" for p in pngs],
           "- `outputs/l1_summary.csv`, `outputs/l1_neighbors.csv`, `outputs/l1_yearly.csv`, `outputs/l1_crash_exits.csv`, `outputs/l1_regimes.csv`, `outputs/l1_equity.csv`",
           ]
    RESULT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[l1] 결과: {RESULT_MD}", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    for c in CANDIDATES[1:]:
        print(c, "합격" if v[c]["pass"] else "불합격", v[c]["reasons"], flush=True)


if __name__ == "__main__":
    main()
