"""실적급등 타이밍 탐색 — 매수 시점 B1~B5 × 보유 20~120거래일 (계획: docs/research/pead_timing_plan.md, 태그 pead-timing-plan).

일회성 리포트 스크립트. 탐색이지 판정이 아니다. 신호·가격·비용·세금은 SW1(scripts/sw1_experiments.py, core/sw1.py),
2022~2026-09 참고는 봉인 ⓐ(scripts/sealed_a_sw1.py) 데이터 그대로. 계산은 core/pead_timing.py(순수 함수).
시작할 때 계획 태그 내용 = 작업 폴더를 확인하고, 계획 6절 점검이 틀리면 멈춘다.

실행: python -u -m scripts.pead_timing
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

from core import pead_timing as pt  # noqa: E402
from core import sealed_a as sa  # noqa: E402
from core import sw1  # noqa: E402
from scripts import sealed_a_b1 as b1  # noqa: E402
from scripts import sealed_a_sw1 as sas  # noqa: E402
from scripts import sealed_a_sw1_data as sd  # noqa: E402
from scripts import sw1_data_check as dc  # noqa: E402
from scripts import sw1_experiments as sx  # noqa: E402

PLAN_TAG = "pead-timing-plan"
PLAN = "docs/research/pead_timing_plan.md"
RESULT_MD = ROOT / "docs" / "research" / "pead_timing_results.md"
CURVE_PNG = ROOT / "docs" / "research" / "pead_timing_curve.png"
OUT_JSON = ROOT / "outputs" / "pead_timing.json"
SEED, ITERS, RATE = 20261111, 1000, 0.22
SW1_B160 = (601, 0.53)  # docs/results/sw1.md
SEALED_B160 = (382, -0.21)  # docs/results/sealed_a_sw1.md
COSTS = sw1.Costs()


def verify_plan() -> str:
    """계획 문서 작업 폴더 = 태그 커밋 내용(LF 기준). 다르면 멈춘다. 출력: 태그 커밋 해시(짧게)."""
    work = sx.lf_sha256((ROOT / PLAN).read_bytes())
    tagged = sx.lf_sha256(subprocess.run(["git", "show", f"{PLAN_TAG}:{PLAN}"], cwd=ROOT, check=True,
                                         capture_output=True).stdout)
    if work != tagged:
        raise SystemExit(f"[pead-timing] 계획 문서가 태그 {PLAN_TAG}와 다름 — 멈춤")
    return subprocess.run(["git", "rev-parse", "--short", f"{PLAN_TAG}^{{commit}}"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def matrices(al: dict[str, pt.Aligned]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """종목 → 행 번호 이름 목록과 시가·종가·lastv 행렬."""
    names = sorted(al)
    return (names, np.vstack([al[t].open for t in names]), np.vstack([al[t].close for t in names]),
            np.vstack([al[t].lastv for t in names]))


def stats(T: pd.DataFrame, periods: tuple[tuple[int, int], ...]) -> dict:
    """매매 표 → 평균·중앙값·승률·구간 평균 (% 단위)."""
    per = [float(T.loc[(T["year"] >= a) & (T["year"] <= b), "ex_after"].mean() * 100) for a, b in periods]
    return {"n": int(len(T)), "mean": float(T["ex_after"].mean() * 100), "median": float(T["ex_after"].median() * 100),
            "win": float((T["ex_after"] > 0).mean() * 100), "ex_pre": float(T["ex_pre"].mean() * 100),
            "p1": per[0], "p2": per[1]}


def run_cell(sig, al, days, bench, fx, mode, hold, pool_rows, M, years, first_year, n_years, periods) -> tuple[dict, pd.DataFrame]:
    """한 칸: 매매 → 수익 → 무작위 1,000회 백분위. pool_rows(buy_day, ticker) → 후보 행 번호 배열."""
    tr, sk = pt.make_trades(sig, al, days, mode, hold)
    T = pt.price_trades(tr, bench, fx, mode, COSTS, RATE)
    pools = [pool_rows(b, t) for b, t in zip(T["buy_day"], T["ticker"])]
    _, O, C, L = M
    cands = pt.candidates(T, pools, O, C, L, bench, fx, years, mode, COSTS)
    rm = sx.random_means(cands, ITERS, SEED, RATE, first_year, n_years)
    S = stats(T, periods)
    S.update(pct=sw1.percentile_of(T["ex_after"].mean(), rm), r_median=float(np.median(rm) * 100),
             skip=sk["skip"].value_counts().to_dict(), cand_min=int(min(len(c["ret"]) for c in cands)))
    return S, T


# ── 2012~2021 ────────────────────────────────────────────────────────────────


def main_period() -> dict:
    qqq = pd.read_parquet(dc.BT_CACHE / "QQQ.parquet")[["open", "close"]]
    if qqq.index[-1] > sx.SEAL:
        raise SystemExit("[pead-timing] QQQ 봉인일 뒤 행 — 멈춤")
    days = qqq.index
    fx = sx.load_fx(days).to_numpy()
    k = sx.rebuild_announcements(days, qqq["close"])
    ref = pd.read_csv(dc.CACHE / "announcements.csv")
    sig = k[k["signal"]].copy()
    if len(sig) != 636 or set(sig["accn"]) != set(ref.loc[ref["signal"], "accn"]):
        raise SystemExit(f"[pead-timing] 신호 재현 불일치 {len(sig)}건 — 멈춤")
    sig["d0"] = pd.to_datetime(sig["d0"])
    sig = sig[["ticker", "d0", "accn", "in_ndx", "in_sp100"]]
    print(f"[pead-timing] 신호 재현 636건 확인", flush=True)

    raw = {t: sx.load_ohlc(t) for t in sx.universe_tickers()}
    al = {t: pt.align(v, days) for t, v in raw.items() if v is not None}
    bench = pt.align(qqq, days)
    M = matrices(al)
    row_of = {t: i for i, t in enumerate(M[0])}
    ndx, sp = dc.ndx_members(), dc.sp100_members()

    def pool_rows(b, t):
        me = (pd.Timestamp(b) - pd.offsets.MonthEnd(1)).normalize()
        return np.array(sorted(row_of[x] for x in (ndx.get(me, set()) | sp.get(me, set())) - {t} if x in row_of), dtype=int)

    years = days.year.to_numpy()
    periods = ((2012, 2016), (2017, 2021))
    cells = []
    for mode in pt.BUY_MODES:
        for hold in pt.HOLDS:
            S, T = run_cell(sig, al, days, bench, fx, mode, hold, pool_rows, M, years, 2012, 11, periods)
            if (mode, hold) == ("B1", 60) and (S["n"], round(S["mean"], 2)) != SW1_B160:
                raise SystemExit(f"[pead-timing] B1·60 SW1 재현 불일치 ({S['n']}건) — 멈춤")
            cells.append({"mode": mode, "hold": hold, **S})
            print(f"[pead-timing] {mode}·{hold} 계산 끝", flush=True)
    print("[pead-timing] B1·60 = SW1 결과(601건) 재현 확인", flush=True)

    curves = {}
    for nm, m in (("신호 전체", slice(None)), ("나스닥100만", sig["in_ndx"].astype(bool)), ("S&P100만", sig["in_sp100"].astype(bool))):
        c, n = pt.cumulative_excess(sig[m], al, bench.close, days)
        curves[nm] = {"curve": (c * 100).tolist(), "n": n}
    gap = {"ann": int(len(k)), "not_judged": int((~k["judged"]).sum()), "signals": int(len(sig)),
           "price_tickers": len(al), "universe": len(raw)}
    return {"cells": cells, "curves": curves, "gap": gap}


# ── 2022 ~ 2026-09 (봉인 ⓐ, 참고) ───────────────────────────────────────────


def sealed_reference(want: list[tuple[str, int]]) -> list[dict]:
    """봉인 ⓐ 데이터로 지정 칸만 (B1·60 + 고른 칸). B1·60은 봉인 ⓐ 결과와 같아야 한다."""
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
    sig = sig.drop_duplicates(["ticker", "d0"])[["ticker", "d0", "sym", "in_ndx", "in_sp100"]]
    al = {c: pt.align(p[["open", "close"]], days) for c, p in prices.items()}
    bench = pt.align(qq, days)
    M = matrices(al)
    row_of = {t: i for i, t in enumerate(M[0])}
    m = sd.members()
    cik_of = {r["ticker"]: str(int(r["cik"])) for _, r in pc[pc["use"]].iterrows()}

    def pool_rows(b, t):
        me = (pd.Timestamp(b) - pd.offsets.MonthEnd(1)).normalize()
        ts = m["ndx"].get(me, set()) | m["sp100"].get(me, set())
        return np.array(sorted(row_of[c] for c in {cik_of[x] for x in ts if x in cik_of} - {t} if c in row_of), dtype=int)

    out = []
    for mode, hold in want:
        S, T = run_cell(sig, al, days, bench, fx.to_numpy(), mode, hold, pool_rows, M, days.year.to_numpy(), 2022, 5,
                        ((2022, 2023), (2024, 2026)))
        if not (T["d0"].min() >= sd.D0_FIRST and T["sell_day"].max() <= sa.SEAL_LAST):
            raise SystemExit("[pead-timing] 봉인 ⓐ 기간 밖 매매 — 멈춤")
        if (mode, hold) == ("B1", 60) and (S["n"], round(S["mean"], 2)) != SEALED_B160:
            raise SystemExit(f"[pead-timing] 2022~ B1·60 봉인 ⓐ 재현 불일치 ({S['n']}건) — 멈춤")
        out.append({"mode": mode, "hold": hold, **S})
    print("[pead-timing] 2022~ B1·60 = 봉인 ⓐ 결과(382건) 재현 확인", flush=True)
    return out


# ── 그림·보고서 ─────────────────────────────────────────────────────────────


def draw_curves(curves: dict) -> None:
    from matplotlib.figure import Figure

    import matplotlib
    from notify.curve_chart import korean_font

    font = korean_font()
    matplotlib.rcParams["axes.unicode_minus"] = False
    if font:
        matplotlib.rcParams["font.family"] = [font, "DejaVu Sans"]
    fig = Figure(figsize=(9, 5), dpi=120)
    ax = fig.subplots()
    x = np.arange(1, 121)
    for (nm, d), col in zip(curves.items(), ("#1f4e79", "#c0504d", "#9bbb59")):
        ax.plot(x, d["curve"], color=col, lw=2 if nm == "신호 전체" else 1.4, label=f"{nm} ({d['n']}건)")
    ax.axhline(0, color="#888", lw=0.8)
    for h in pt.HOLDS:
        ax.axvline(h, color="#ddd", lw=0.6, zorder=0)
    ax.set_xlabel("반응일 뒤 거래일")
    ax.set_ylabel("QQQ 대비 평균 누적 초과 (%, 비용 전)")
    ax.set_title("실적급등 신호 뒤 평균 누적 초과 — 2012~2021, 반응일 종가 기준")
    ax.legend(loc="best", frameon=False)
    ax.grid(axis="y", color="#eee")
    fig.tight_layout()
    fig.savefig(CURVE_PNG)


def p(x: float, d: int = 2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{d}f}%"


def write_report(res: dict, chosen, notes, ref: list[dict], plan_commit: str) -> None:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    g = pd.DataFrame(res["cells"]).set_index(["mode", "hold"])
    nt = {(x["mode"], x["hold"]): x for x in notes}
    labels = {"B1": "B1 다음 날 시가", "B2": "B2 다음 날 종가", "B3": "B3 +3일 종가", "B4": "B4 +5일 종가", "B5": "B5 되돌림"}

    def table(col: str, fmt) -> list[str]:
        L = ["| 매수 \\ 보유 | " + " | ".join(f"{h}일" for h in pt.HOLDS) + " |", "| --- |" + " --- |" * len(pt.HOLDS)]
        for m in pt.BUY_MODES:
            L.append(f"| {labels[m]} | " + " | ".join(fmt(g.loc[(m, h)], (m, h)) for h in pt.HOLDS) + " |")
        return L

    def heat(r, key):
        v = r["mean"]
        mark = "🟩" if v >= 0.5 else ("🟢" if v > 0 else ("🔴" if v <= -0.5 else "🟠"))
        s = f"{mark} {p(v)}"
        if chosen is not None and key == tuple(chosen):
            s = f"**{s} ◀**"
        elif key == ("B1", 60):
            s = f"{s} (기준)"
        return s

    cg = res["gap"]
    sel_txt = (f"**{labels[chosen[0]]} · {chosen[1]}거래일 보유**" if chosen is not None else "**해당 칸 없음 → 기준 B1·60 유지**")
    L = ["# 실적급등 타이밍 탐색 — 결과", "",
         f"> 계획: `{PLAN}` (태그 `{PLAN_TAG}` = 커밋 `{plan_commit}`, 실행 시 작업 폴더와 같음 확인) · 실행 코드 커밋 `{head}` · 계산 1회",
         "> 탐색이지 판정이 아니다. 25칸을 한 번에 본 결과라 가장 좋은 칸의 숫자는 위로 치우쳐 있다.", "",
         "- 신호: SW1 그대로(나스닥100 ∪ S&P100, 반응일 +5%·QQQ 대비 +5%p·거래량 2배), 반응일 2012~2021, 636건 재현 확인",
         "- 초과 = 신호 종목 − 같은 날·같은 가격 종류로 사고판 QQQ, 원화 비용(수수료 0.07%·환전 0.1% 편도)·세금(매도 연도 장부 순이익 22%) 뒤",
         f"- 무작위: 같은 매수일·같은 예정 매도일, 매수일 전 달 말 명단 무작위 1종목, {ITERS}회, 시드 {SEED}",
         "- 점검: B1·60 = SW1 결과(601건, +0.53%), 2022~ B1·60 = 봉인 ⓐ 결과(382건, −0.21%) 재현 확인", "",
         f"## 선택: {sel_txt}", ""]
    ok = [x for x in notes if x["ok"]]
    both = [x for x in notes if x["both_periods"]]
    L += [f"- 1단계(두 구간 모두 > 0): {len(both)}칸 — " + (", ".join(f"{x['mode']}·{x['hold']}" for x in both) or "없음"),
          f"- 2단계(+ 상하좌우 이웃 평균 > 0): {len(ok)}칸 — " + (", ".join(f"{x['mode']}·{x['hold']} ({x['pct']:.1f}백분위)" for x in
                                                           sorted(ok, key=lambda x: -x["pct"])) or "없음")]
    if chosen is not None:
        r = g.loc[tuple(chosen)]
        L.append(f"- 3단계: 그중 무작위 백분위 최고 → {chosen[0]}·{chosen[1]} ({r['pct']:.1f}백분위, 평균 {p(r['mean'])}, "
                 f"2012~16 {p(r['p1'])} / 2017~21 {p(r['p2'])}, 매매 {int(r['n'])})")
    L += ["", "## 평균 초과 (비용·세금 뒤, 매매당) — 히트맵", "",
          "🟩 ≥ +0.5% · 🟢 0 ~ +0.5% · 🟠 −0.5% ~ 0 · 🔴 ≤ −0.5% · ◀ 선택", ""]
    L += table("mean", heat)
    L += ["", "## 무작위 1,000회 백분위", ""]
    L += table("pct", lambda r, k: f"{r['pct']:.1f}")
    L += ["", "## 2012~2016 / 2017~2021 평균 초과", ""]
    L += table("p", lambda r, k: f"{p(r['p1'])} / {p(r['p2'])}")
    L += ["", "## 중앙값 초과 · 승률", ""]
    L += table("med", lambda r, k: f"{p(r['median'])} · {r['win']:.0f}%")
    L += ["", "## 매매 수 · 평균 초과 비용 전(달러)", ""]
    L += table("n", lambda r, k: f"{int(r['n'])} · {p(r['ex_pre'])}")
    L += ["", "## 고르는 규칙 칸별 판단", "", "✔ 두 구간 > 0 · ◇ 이웃 평균 모두 > 0", ""]
    L += table("rule", lambda r, k: ("✔" if nt[k]["both_periods"] else "·") + ("◇" if nt[k]["neighbors"] else "·"))
    L += ["", "## 평균 누적 초과 곡선 (참고, 선택에 안 씀)", "",
          "![평균 누적 초과](pead_timing_curve.png)", "",
          "| 반응일 뒤 | " + " | ".join(res["curves"]) + " |", "| --- |" + " --- |" * len(res["curves"])]
    for kk in (1, 3, 5, 10, 20, 40, 60, 90, 120):
        L.append(f"| {kk}일 | " + " | ".join(p(d["curve"][kk - 1]) for d in res["curves"].values()) + " |")
    L += ["", "- 신호 단위(겹침 규칙 없음), 반응일 + 120거래일이 2021-12-31 안인 신호, 비용 전 달러. 나스닥100·S&P100 양쪽 명단이면 둘 다에 들어감."]
    L += ["", "## 2022 ~ 2026-09 (봉인 ⓐ, 이미 개봉) — 참고만", "",
          "| 칸 | 매매 | 평균 초과(세금 뒤) | 중앙값 | 승률 | 2022~23 / 2024~26 | 무작위 백분위 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in ref:
        L.append(f"| {r['mode']}·{r['hold']} | {r['n']} | {p(r['mean'])} | {p(r['median'])} | {r['win']:.1f}% | "
                 f"{p(r['p1'])} / {p(r['p2'])} | {r['pct']:.1f} |")
    L += ["", "- 같은 방법(봉인 ⓐ 데이터·매도 예정일 2026-09-30 이내). 계획 9절대로 이 두 칸 말고는 계산하지 않았다."]
    L += ["", "## 데이터 공백", "", "| 항목 | 값 |", "| --- | --- |",
          f"| 실적 발표 중 판정 못 함 (가격·시각·이력 부족, SW1과 같음) | {cg['not_judged']}/{cg['ann']} ({cg['not_judged'] / cg['ann']:.1%}) |",
          f"| 명단 티커 중 가격 있음 | {cg['price_tickers']}/{cg['universe']} |",
          "| 종목·달 범위 (계획서 SW1 2장) | 나스닥100 88.5%, S&P 100 97.5% |", "",
          "칸마다 매매 안 된 신호 (636건 중):", "",
          "| 칸 | 매매 | 안 됨 | 사유 |", "| --- | --- | --- | --- |"]
    for m in pt.BUY_MODES:
        for h in pt.HOLDS:
            r = g.loc[(m, h)]
            sk = r["skip"]
            L.append(f"| {m}·{h} | {int(r['n'])} | {sum(sk.values())} | " + ", ".join(f"{a} {b}" for a, b in sk.items()) + " |")
    L += ["", "## 결과를 본 뒤 바꾼 점", "", "없음", ""]
    RESULT_MD.write_text("\n".join(L), encoding="utf-8", newline="\n")


def main() -> None:
    plan_commit = verify_plan()
    print(f"[pead-timing] 계획 태그 확인 ({plan_commit})", flush=True)
    res = main_period()
    grid = pd.DataFrame(res["cells"])[["mode", "hold", "mean", "p1", "p2", "pct"]]
    chosen, notes = pt.select_cell(grid)
    want = [("B1", 60)] + ([chosen] if chosen is not None and tuple(chosen) != ("B1", 60) else [])
    ref = sealed_reference(want)
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps({**res, "chosen": chosen, "notes": notes, "ref": ref}, ensure_ascii=False, indent=1,
                                   default=str), encoding="utf-8")
    draw_curves(res["curves"])
    write_report(res, chosen, notes, ref, plan_commit)
    print(f"[pead-timing] 완료 → {RESULT_MD.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
