"""사이클 실험실 1단계 — 실행기 (docs/prereg/cycle_lab_discovery.md, 태그 cycle-lab-discovery-prereg).

실행:
  python -m scripts.cycle_lab.run --fetch            # 데이터 받기(태그 push 뒤에만) + 계산
  python -m scripts.cycle_lab.run --pushed-at <시각>  # 캐시로 계산
출력: docs/results/cycle_lab/v1/ — 이미 있으면 덮어쓰지 않고 멈춘다.
결과를 본 뒤 규칙을 바꾸지 않는다. 2013~2021 계산은 하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.cycle_lab import data as dm  # noqa: E402
from scripts.cycle_lab import lab  # noqa: E402
from scripts.cycle_lab.gate import CUTOFF, assert_cutoff  # noqa: E402

OUT = ROOT / "docs" / "results" / "cycle_lab" / "v1"
RUN_LOG = dm.CACHE / "run_log.json"
TAG = "cycle-lab-discovery-prereg"
M1_PAIRS = {"Chips": "A34SNO", "Hardw": "A34SNO", "ElcEq": "A34SNO", "Cnstr": "PERMIT", "BldMt": "PERMIT",
            "Autos": "TOTALSA", "Rubbr": "TOTALSA", "Trans": "TSIFRGHT", "Aero": "TSIFRGHT", "Oil": "WTISPLC",
            "Coal": "WTISPLC", "Mines": "WPU102301", "Steel": "WPU102301", "Mach": "DGORDER"}
CONTAMINATED = [("Chips", "M1"), ("Hardw", "M1"), ("ElcEq", "M1")]
METHODS = ["M1", "M2", "M3", "M4", "M5", "M6"]
EVENT = {"M1", "M2", "M3", "M5"}
QQQ_FROM = pd.Timestamp("1999-04-30")


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_all() -> None:
    """모든 데이터 받기. 첫 수신 시각을 run_log에 남긴다."""
    dm.CACHE.mkdir(parents=True, exist_ok=True)
    state = json.loads(RUN_LOG.read_text(encoding="utf-8")) if RUN_LOG.exists() else {}
    failures = []
    jobs = [("french", k) for k in dm.FRENCH_FILES] + [("alfred", s) for s in dm.ALFRED_IDS] + \
           [("current", s) for s in dm.CURRENT_IDS] + [("yf", t) for t in dm.YF_TICKERS]
    for kind, name in jobs:
        try:
            fn = {"french": dm.fetch_french, "alfred": dm.fetch_alfred, "current": dm.fetch_current, "yf": dm.fetch_yf}[kind]
            meta = fn(name)
            if "first_data_received_utc" not in state:
                state["first_data_received_utc"] = meta["fetched_utc"]
                RUN_LOG.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
            log(f"받음 {kind} {name}: {meta['first']} ~ {meta['last']} ({meta['rows']}행)")
        except Exception as exc:  # noqa: BLE001 — 실패 목록을 모아 보고
            failures.append(f"{kind} {name}: {exc}")
            log(f"실패 {kind} {name}: {exc}")
    if failures:
        raise SystemExit("BLOCKED — 데이터 수신 실패: " + "; ".join(failures))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def fmt(x, pct=True, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return f"{x * 100:.{d}f}%" if pct else f"{x:.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and not np.isfinite(v)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--pushed-at", default="")
    args = ap.parse_args()

    if OUT.exists():
        raise SystemExit(f"BLOCKED — {OUT.relative_to(ROOT)} 이미 있음(덮어쓰기 금지)")
    if args.fetch:
        fetch_all()

    # ── 읽기 + 날짜 assert ─────────────────────────────────────────────────
    ind_all = dm.load_french("ind49")
    ff = dm.load_french("ff3")
    usrec = dm.load_current("USREC")
    alfred = {s: dm.load_alfred(s) for s in dm.ALFRED_IDS}
    current = {s: dm.load_current(s) for s in dm.ALFRED_IDS}
    qqq, ndx = dm.load_yf("QQQ"), dm.load_yf("^NDX")
    maxdates = {"french_ind49": ind_all.index.max(), "french_ff3": ff.index.max(), "USREC": usrec.index.max(),
                "QQQ": qqq.index.max(), "^NDX": ndx.index.max()}
    for s in dm.ALFRED_IDS:
        a = alfred[s]
        maxdates[f"alfred_{s}(obs)"] = a["obs"].max() if len(a) else pd.NaT
        maxdates[f"alfred_{s}(vintage)"] = a["realtime_end"].max() if len(a) else pd.NaT
        maxdates[f"fred_{s}"] = current[s].index.max()
    bad = {k: v for k, v in maxdates.items() if pd.notna(v) and pd.Timestamp(v) > CUTOFF}
    assert not bad, f"2012-12-31 뒤 날짜: {bad}"
    overall_max = max(pd.Timestamp(v) for v in maxdates.values() if pd.notna(v))
    log(f"입력 최대 날짜 assert 통과: 전체 최대 {overall_max.date()} ≤ 2012-12-31 ({len(maxdates)}개 입력)")

    # ── 산업 정리 ─────────────────────────────────────────────────────────
    win = ind_all.loc[lab.EVAL_START:lab.EVAL_END]
    excluded = [c for c in ind_all.columns if win[c].isna().any()]
    ind = ind_all[[c for c in ind_all.columns if c not in excluded]]
    order = {c: i for i, c in enumerate(ind.columns)}
    mkt_rf, rf = ff["Mkt-RF"], ff["RF"]
    market = mkt_rf + rf
    sel = lab.select_cycle(ind, mkt_rf, rf, usrec)
    cycle = [c for c in ind.columns if sel.loc[c, "cycle"]]
    log(f"사이클 산업 {len(cycle)}개: {', '.join(cycle)} (제외 {excluded})")

    # ── 핵심 ──────────────────────────────────────────────────────────────
    qm, n_fixed = dm.qqq_monthly_total(qqq, ndx)
    core = market.loc[:QQQ_FROM - pd.Timedelta(days=1)]
    q = qm.loc[QQQ_FROM:lab.EVAL_END]
    exp_q = pd.date_range(QQQ_FROM, lab.EVAL_END, freq="ME")
    if len(q) != len(exp_q) or q.isna().any():
        raise SystemExit("BLOCKED — QQQ 월간 총수익 결측")
    core = pd.concat([core, q]).sort_index()

    # ── 실시간 지표 ───────────────────────────────────────────────────────
    judge = pd.date_range("1946-01-31", lab.EVAL_END, freq="ME")
    rt: dict[tuple[str, tuple], pd.DataFrame] = {}
    for s, lags in [(s, (2, 8)) for s in set(M1_PAIRS.values())] + [("BAA", (2,)), ("GS10", (2,)), ("TB3MS", (2,)),
                                                                     ("INDPRO", (2, 8)), ("GS10", (2, 8)), ("TB3MS", (2, 8))]:
        rt[(s, lags)] = lab.realtime_values(alfred[s], current[s], judge, lags)
    fallback = []
    use = {("BAA", (2,)): "M2", ("GS10", (2,)): "M2·M3", ("TB3MS", (2,)): "M3", ("INDPRO", (2, 8)): "M4",
           ("GS10", (2, 8)): "M4", ("TB3MS", (2, 8)): "M4"}
    for (s, lags), t in rt.items():
        lv = t.loc[pd.Timestamp("1963-06-30"):, "level"].dropna()
        fallback.append({"series": s, "lags": "t-" + ",t-".join(map(str, lags)), "use": use.get((s, lags), "M1"),
                         "months": len(lv), "first_judge": str(lv.index.min().date()) if len(lv) else "",
                         "level1": float((lv == 1).mean()) if len(lv) else np.nan,
                         "level2": float((lv == 2).mean()) if len(lv) else np.nan,
                         "level3": float((lv == 3).mean()) if len(lv) else np.nan})
    fallback = pd.DataFrame(fallback)

    sig_tab: dict[tuple[str, str], pd.DataFrame] = {}
    for c in cycle:
        if c in M1_PAIRS:
            sig_tab[(c, "M1")] = lab.m1_signal(lab.pct_change_rt(rt[(M1_PAIRS[c], (2, 8))]))
    m2 = lab.m2_signal(rt[("BAA", (2,))]["v2"] - rt[("GS10", (2,))]["v2"])
    m3 = lab.m3_signal(rt[("GS10", (2,))]["v2"] - rt[("TB3MS", (2,))]["v2"])
    g, t3, ip = rt[("GS10", (2, 8))], rt[("TB3MS", (2, 8))], rt[("INDPRO", (2, 8))]
    phase = lab.m4_phase(ip["v2"], ip["v8"], g["v2"] - t3["v2"], g["v8"] - t3["v8"])
    cyc_ret = ind[cycle]
    m6 = lab.m6_state(cyc_ret)
    m6ok = lab.m6_ok(cyc_ret)
    m4_fixed = {}
    for c in cycle:
        sig_tab[(c, "M2")], sig_tab[(c, "M3")] = m2, m3
        best, means = lab.m4_fix(phase, (ind[c] - market).dropna())
        m4_fixed[c] = (best, means)
        st = (phase == best) if best is not None else pd.Series(False, index=phase.index)
        st = st & phase.notna() & (phase.index >= lab.M4_SIG_FIRST)
        sig_tab[(c, "M4")] = pd.DataFrame({"sig": st, "ok": phase.notna() & (phase.index >= lab.M4_SIG_FIRST)})
        r = ind[c].loc[ind[c].first_valid_index():]
        if r.isna().any():
            r = r.loc[r[r.isna()].index.max() + pd.offsets.MonthEnd(1):]
        sig_tab[(c, "M5")] = lab.m5_signal(r)
        sig_tab[(c, "M6")] = pd.DataFrame({"sig": m6[c], "ok": m6ok[c]})

    # ── 칸 계산 ───────────────────────────────────────────────────────────
    rows = []
    for c in cycle:
        level = (1.0 + ind[c].loc[ind[c].first_valid_index():].fillna(0.0)).cumprod()
        troughs_all = lab.zigzag_troughs(level)
        for m in METHODS:
            if (c, m) not in sig_tab:
                continue
            st = sig_tab[(c, m)]
            okm = st.index[st["ok"].astype(bool)]
            if not len(okm):
                continue
            r0 = lab.M4_SIG_FIRST + pd.offsets.MonthEnd(1) if m == "M4" else max(lab.EVAL_START, lab.add_months(okm.min(), 1))
            months = pd.date_range(r0, lab.EVAL_END, freq="ME")
            sig = st["sig"].astype(bool)
            w, eps = (lab.event_weights if m in EVENT else lab.state_weights)(sig, months)
            res = lab.evaluate_cell(w, eps, core, ind[c])
            jm = pd.date_range(lab.add_months(r0, -1), lab.add_months(lab.EVAL_END, -1), freq="ME")
            sj = sig.reindex(jm).fillna(False).astype(bool)
            raw = int(sj.sum()) if m in EVENT else int((sj & ~sj.shift(1, fill_value=False)).sum())
            troughs = [b for b in troughs_all if r0 <= b <= lab.EVAL_END]
            ls = lab.lead_stats([e["signal"] for e in eps], troughs, jm)
            rows.append({"industry": c, "method": m, "indicator": M1_PAIRS.get(c, "") if m == "M1" else
                         {"M2": "BAA−GS10", "M3": "GS10−TB3MS", "M4": "INDPRO×곡선", "M5": "산업지수", "M6": "12-1 수익"}[m],
                         **res, "raw_signals": raw, "signals_per_year": raw / (len(jm) / 12.0),
                         "troughs": len(troughs), **{f"lead_{k}": v for k, v in ls.items()},
                         "p8_cash10": "아니오", "p8_hold_lt60d": "예" if (res["min_hold"] == 1) else "아니오",
                         "p8_turnover100": "예" if res["turnover"] > 1.0 else "아니오", "p8_largecap_stop": "아니오",
                         "p8_crowding": "판정 안 함", "ind_order": order[c], "method_order": METHODS.index(m),
                         "m4_phase": lab.PHASES[m4_fixed[c][0]] if (m == "M4" and m4_fixed[c][0] is not None) else ""})
            log(f"칸 {c}×{m}: 진입 {res['entries']} 초과 {fmt(res['excess_cagr'])} 백분위 {res['pctile']}")
    cells = pd.DataFrame(rows)
    cells["q_bh"] = lab.bh_q(cells["p"].to_numpy())
    cells["bh10_pass"] = cells["q_bh"] <= 0.10
    cells["eligible"] = (cells["pctile"] >= 90) & (cells["excess_cagr"] > 0) & (cells["entries"] >= 8)
    N = len(cells)
    top = lab.pick_top(cells)
    top_x = lab.pick_top(cells, exclude=CONTAMINATED)

    # ── 산출물 ────────────────────────────────────────────────────────────
    OUT.mkdir(parents=True, exist_ok=False)
    sel_out = sel.copy()
    sel_out.index.name = "industry"
    sel_out.to_csv(OUT / "cycle_industries.csv", encoding="utf-8-sig")
    cells_out = cells.copy()
    for col in ("eval_start", "eval_end"):
        cells_out[col] = pd.to_datetime(cells_out[col]).dt.strftime("%Y-%m")
    cells_out.drop(columns=["ind_order", "method_order"]).to_csv(OUT / "cells.csv", index=False, encoding="utf-8-sig")

    man = pd.DataFrame(dm.manifest_rows())
    man["series"] = man["name"].str.replace(r"^(alfred|fred|french|yf)_", "", regex=True)
    fb_txt = {s: "; ".join(f"{r.use}({r.lags}) 1순위 {fmt(r.level1, d=1)}·2순위 {fmt(r.level2, d=1)}·3순위 {fmt(r.level3, d=1)}"
                           for r in g.itertuples()) for s, g in fallback.groupby("series")}
    man["fallback"] = [fb_txt.get(s, "") if n.startswith("alfred_") else "" for s, n in zip(man["series"], man["name"])]
    man[["name", "series", "source", "first", "last", "rows", "sha256", "fetched_utc", "cutoff", "fallback"]].to_csv(
        OUT / "data_manifest.csv", index=False, encoding="utf-8-sig")

    _heatmap(cells, cycle)

    run_state = json.loads(RUN_LOG.read_text(encoding="utf-8")) if RUN_LOG.exists() else {}
    tag_commit = git("rev-list", "-n", "1", TAG)
    commit_time = git("log", "-1", "--format=%cI", TAG) if tag_commit else "(태그 없음)"
    elig_by_m = cells.groupby("method")["eligible"].sum().reindex(METHODS).fillna(0).astype(int)

    def top_md(t: pd.DataFrame) -> str:
        if t.empty:
            return "(자격 통과 칸 없음 — 0개)"
        show = pd.DataFrame({"순위": range(1, len(t) + 1), "산업": t["industry"].values, "예측법": t["method"].values,
                             "지표": t["indicator"].values, "평가 시작": pd.to_datetime(t["eval_start"]).dt.strftime("%Y-%m").values,
                             "진입": t["entries"].values, "무작위 백분위": [f"{x:.1f}" for x in t["pctile"]],
                             "p": [f"{x:.3f}" for x in t["p"]], "q(BH)": [f"{x:.3f}" for x in t["q_bh"]],
                             "초과수익": [fmt(x) for x in t["excess_cagr"]], "낙폭 차이": [fmt(x) for x in t["mdd_diff"]],
                             "상시 보유 대비": [fmt(x) for x in t["vs_constant"]],
                             "비고": ["대조군" if m == "M6" else ("오염 가능(8절)" if (i, m) in CONTAMINATED else "")
                                    for i, m in zip(t["industry"], t["method"])]})
        return md_table(show)

    same = (list(zip(top.get("industry", []), top.get("method", []))) == list(zip(top_x.get("industry", []), top_x.get("method", []))))
    top_text = "\n".join([
        "# 사이클 실험실 1단계 — 상위 3개", "",
        f"사전 등록: `docs/prereg/cycle_lab_discovery.md` (태그 `{TAG}`). 자격: 무작위 ≥90백분위·초과수익 >0·진입 ≥8, 같은 예측법 최대 1개.", "",
        "## 공식 선정 (전체 순위)", "", top_md(top), "",
        "## 참고: Chips·Hardw·ElcEq의 M1(A34SNO) 칸을 뺀 순위", "", top_md(top_x), "",
        "두 결과가 " + ("같다." if same else "**다르다 — 사용자 결정 필요**(사전 등록 7·8절)."), ""])
    (OUT / "top3.md").write_text(top_text, encoding="utf-8")

    show_cells = cells.sort_values(["ind_order", "method_order"])
    tbl = pd.DataFrame({"산업": show_cells["industry"], "예측법": show_cells["method"], "지표": show_cells["indicator"],
                        "평가 시작": pd.to_datetime(show_cells["eval_start"]).dt.strftime("%Y-%m"),
                        "진입": show_cells["entries"], "원 신호": show_cells["raw_signals"],
                        "신호/년": [f"{x:.2f}" for x in show_cells["signals_per_year"]],
                        "초과수익": [fmt(x) for x in show_cells["excess_cagr"]],
                        "낙폭 차이": [fmt(x) for x in show_cells["mdd_diff"]],
                        "평균 비중": [fmt(x, d=1) for x in show_cells["avg_weight"]],
                        "회전/년": [fmt(x, d=0) for x in show_cells["turnover"]],
                        "평균 보유(월)": [fmt(x, pct=False, d=1) for x in show_cells["avg_hold"]],
                        "상시 보유 대비": [fmt(x) for x in show_cells["vs_constant"]],
                        "백분위": [("" if not np.isfinite(x) else f"{x:.1f}") for x in show_cells["pctile"]],
                        "p": [f"{x:.3f}" for x in show_cells["p"]], "q": [f"{x:.3f}" for x in show_cells["q_bh"]],
                        "자격": ["O" if e else "" for e in show_cells["eligible"]]})
    lead_tbl = pd.DataFrame({"산업": show_cells["industry"], "예측법": show_cells["method"], "바닥 수": show_cells["troughs"],
                             "적중": show_cells["lead_hits"], "적중률": [fmt(x, d=0) for x in show_cells["lead_hit_rate"]],
                             "임의 달 기준선": [fmt(x, d=0) for x in show_cells["lead_base_rate"]],
                             "선행 중앙값(월)": [fmt(x, pct=False, d=1) for x in show_cells["lead_lead_median"]],
                             "헛신호": show_cells["lead_false"]})
    p8_tbl = pd.DataFrame({"산업": show_cells["industry"], "예측법": show_cells["method"], "현금>10%": show_cells["p8_cash10"],
                           "보유<60일": show_cells["p8_hold_lt60d"], "회전>100%": show_cells["p8_turnover100"],
                           "대형주 손절": show_cells["p8_largecap_stop"], "쏠림 역행": show_cells["p8_crowding"]})
    sel_tbl = sel.loc[cycle]
    sel_show = pd.DataFrame({"산업": cycle, "베타": [f"{x:.2f}" for x in sel_tbl["beta"]],
                             "표준편차 순위": sel_tbl["rank_std"].astype(int).values,
                             "침체기 (산업−시장) 월평균": [fmt(x) for x in sel_tbl["rec_rel"]],
                             "A": ["O" if x else "" for x in sel_tbl["A"]], "B": ["O" if x else "" for x in sel_tbl["B"]]})
    m4_show = pd.DataFrame({"산업": cycle, "고정 국면": [lab.PHASES[m4_fixed[c][0]] if m4_fixed[c][0] is not None else "" for c in cycle],
                            **{p: [fmt(m4_fixed[c][1][p]) for c in cycle] for p in lab.PHASES}})
    fb_show = pd.DataFrame({"시리즈": fallback["series"], "쓰임": fallback["use"], "관측": fallback["lags"],
                            "판정 달 수": fallback["months"], "첫 판정 달": fallback["first_judge"],
                            "1순위(당시 발표본)": [fmt(x, d=0) for x in fallback["level1"]],
                            "2순위(2012-12-31 vintage)": [fmt(x, d=0) for x in fallback["level2"]],
                            "3순위(현재값)": [fmt(x, d=0) for x in fallback["level3"]]})
    report = "\n".join([
        "# 사이클 산업 예측 실험실 — 1단계(발견) 보고서", "",
        "연구 작업이며 투자 조언이 아니다. 달러·세전 순위용(세후 원화 판정은 2단계). 2013-01-01 이후 데이터는 받지도 계산하지도 않았다.", "",
        "## 요약", "",
        f"- 사이클 산업 {len(cycle)}개: {', '.join(cycle)}",
        f"- 결측으로 제외한 산업: {', '.join(excluded) if excluded else '없음'} (남은 산업 n = {ind.shape[1]})",
        f"- 칸 수 N = {N}",
        "- 예측법별 자격 통과 수: " + ", ".join(f"{m} {elig_by_m[m]}" for m in METHODS),
        f"- 공식 상위 {len(top)}개: " + ("; ".join(f"{r.industry}×{r.method} (백분위 {r.pctile:.1f}, 초과 {fmt(r.excess_cagr)}, 진입 {r.entries})"
                                                  for r in top.itertuples()) if len(top) else "없음 → 계열 탈락"),
        f"- 반도체 M1 제외 상위 {len(top_x)}개: " + ("; ".join(f"{r.industry}×{r.method} (백분위 {r.pctile:.1f}, 초과 {fmt(r.excess_cagr)})"
                                                         for r in top_x.itertuples()) if len(top_x) else "없음")
        + (" — 공식과 같음" if same else " — **공식과 다름, 사용자 결정 필요**"),
        f"- BH q ≤ 0.10 칸: {int(cells['bh10_pass'].sum())}개 / {N}",
        "",
        "## 순서 증빙", "",
        f"- 사전 등록 커밋(태그 `{TAG}` = {tag_commit[:7] or '없음'}) 시각: {commit_time}",
        f"- push 시각: {args.pushed_at or '(인자 없음)'}",
        f"- 첫 데이터 수신 시각(UTC): {run_state.get('first_data_received_utc', '(기록 없음)')}",
        f"- 입력 최대 날짜 assert: 통과 — 전체 최대 {overall_max.date()} ≤ 2012-12-31 (입력 {len(maxdates)}개)",
        "",
        "## 사이클 산업 (1963-07 ~ 2012-12)", "",
        f"조건: A 베타 > 1.1, B 표준편차 상위 ⌊n/3⌋ = {ind.shape[1] // 3}, C 침체기 (산업−시장) 하위 ⌊n/2⌋ = {ind.shape[1] // 2}. 사이클 = (A 또는 B) 그리고 C. 전체 표는 `cycle_industries.csv`.", "",
        md_table(sel_show), "",
        "## 칸별 성적", "",
        "초과수익 = 전략 연복리 − 핵심 100% 연복리(같은 창). 낙폭 차이 = 전략 MDD − 핵심 MDD(양수 = 얕음). 상시 보유 대비 = 전략 − 같은 평균 비중 상시 보유.",
        "M2·M3는 모든 사이클 산업에 같은 신호. M4는 1988-01 판정부터(앞 구간은 국면 고정용). M6는 대조군.", "",
        md_table(tbl), "",
        "## M4 고정 국면 (판정 1963-07 ~ 1987-11 → 다음 달 산업−시장 평균)", "", md_table(m4_show), "",
        "## 대체 비율 (판정 1963-06 이후, 값이 있는 달 기준)", "", md_table(fb_show), "",
        f"QQQ 월간 총수익: ^NDX 대비 2%p 넘게 다른 날 {n_fixed}일을 ^NDX 수익률로 바꿈. 핵심 = 1999-03까지 Mkt-RF+RF, 1999-04부터 QQQ.", "",
        "## 선행성 (보고용 — 지그재그 −30%/+40%는 사후 표시)", "", md_table(lead_tbl), "",
        "## P8 점검 (보고용)", "", "쏠림 역행은 저장소에 정의 문서가 없어 판정하지 않았다(사전 등록 7절).", "", md_table(p8_tbl), "",
        "## 사전 등록 8절 이력 (다시 적음)", "",
        "- A34SNO 6개월 변화는 이전 마이크론 1990~2012 해부에서 뽑힌 지표다(발견 구간과 겹침). Chips·Hardw·ElcEq × M1은 오염 가능.",
        "- 2013~2021은 다른 연구에서 쓰인 구간이다. 고정 임계값 +2%p는 변동 큰 지표에서 쉽게 충족될 수 있다(위 신호/년 참고).",
        "- 학계 기대: 산업 순환 이득은 작고 실시간에서 대부분 사라짐(Stangl·Jacobsen·Visaltanachoti 2009 등). 사전 통과 가능성 15~20%.", "",
        "## 파일", "", "`cycle_industries.csv`, `cells.csv`, `heatmap.png`, `top3.md`, `data_manifest.csv`(캐시 meta + 대체 비율 표)", ""])
    (OUT / "report.md").write_text(report, encoding="utf-8")
    summary = {"N": N, "cycle": cycle, "excluded": excluded, "top": top[["industry", "method"]].values.tolist() if len(top) else [],
               "top_excl": top_x[["industry", "method"]].values.tolist() if len(top_x) else [], "same": same,
               "eligible_by_method": {m: int(elig_by_m[m]) for m in METHODS}}
    log(json.dumps(summary, ensure_ascii=False))


def _heatmap(cells: pd.DataFrame, cycle: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    P = cells.pivot(index="industry", columns="method", values="pctile").reindex(index=cycle, columns=METHODS)
    E = cells.pivot(index="industry", columns="method", values="excess_cagr").reindex(index=cycle, columns=METHODS)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(cycle) + 1.5))
    im = ax.imshow(P.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(METHODS)), METHODS)
    ax.set_yticks(range(len(cycle)), cycle)
    for i in range(len(cycle)):
        for j in range(len(METHODS)):
            e = E.iat[i, j]
            if np.isfinite(e):
                ax.text(j, i, f"{e * 100:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if (not np.isfinite(P.iat[i, j]) or P.iat[i, j] < 60) else "black")
    fig.colorbar(im, ax=ax, label="무작위 대조 백분위")
    ax.set_title("사이클 산업 × 예측법 (색: 백분위, 글자: 위성 초과수익 %p, 1963-07~2012-12)")
    fig.tight_layout()
    fig.savefig(OUT / "heatmap.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
