"""P6-1 실험 스크립트 — 포트폴리오 엔진 + 1999~2021 종목 없는 조합(P0·P3·P4·P5).

일회성 리포트 스크립트 (CLAUDE.md scripts/). engine/portfolio.py의 순수 시뮬레이션
함수를 그대로 재사용하고, 여기서는 기간·후보·주변값 조합을 오케스트레이션만 한다.
판정 기준은 configs/p6_1_preregistration.yaml에 이미 커밋되어 있다(결과를 보기 전).

실행:
    python -u -m scripts.p6_1_experiments
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from engine import backtest as bt  # noqa: E402
from engine import portfolio as pf  # noqa: E402

OUT_ROOT = bt.OUTPUT_ROOT / "p6_1"
PREREG_PATH = ROOT / "configs" / "p6_1_preregistration.yaml"
CHECKPOINT_DIR = bt.OUTPUT_ROOT / "p6_1_checkpoint"

FULL_START = date(1999, 3, 10)
FULL_END = date(2021, 12, 31)
SUB_PERIODS = {
    "sub_1999_2007": (date(1999, 3, 10), date(2007, 12, 31)),
    "sub_2008_2015": (date(2008, 1, 1), date(2015, 12, 31)),
    "sub_2016_2021": (date(2016, 1, 1), date(2021, 12, 31)),
}
REPRO_START, REPRO_END = date(2015, 1, 1), date(2021, 12, 31)  # P5-5 QQQ 재현 확인용(원 지시문 그대로)
P5_5_QQQ_REFERENCE = {"posttax_a_pct": 20.60, "posttax_b_pct": 23.62, "mdd_pct": -27.31}

CRASH_WINDOWS = {
    "닷컴 붕괴(2000-03~2002-10)": (date(2000, 3, 1), date(2002, 10, 31)),
    "금융위기(2007-10~2009-03)": (date(2007, 10, 1), date(2009, 3, 31)),
    "코로나(2020-02~2020-03)": (date(2020, 2, 1), date(2020, 3, 31)),
}

MAIN_DRAWDOWN_TRIGGERS = (-20.0, -30.0, -40.0)
NEIGHBOR_DRAWDOWN_TRIGGERS = [(-15.0, -25.0, -35.0), (-20.0, -30.0, -40.0), (-25.0, -35.0, -45.0)]
MAIN_SMA_DAYS = 200
NEIGHBOR_SMA_DAYS = [190, 200, 210]


def _ckpt_path(name: str) -> Path:
    return CHECKPOINT_DIR / f"{name}.pkl"


def _load_ckpt(name: str):
    path = _ckpt_path(name)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_ckpt(name: str, obj) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _ckpt_path(name)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)


def _run_checkpointed(name: str, compute_fn):
    cached = _load_ckpt(name)
    if cached is not None:
        print(f"[checkpoint] {name} <- 불러옴", flush=True)
        return cached
    print(f"[checkpoint] {name} 계산 중 ...", flush=True)
    result = compute_fn()
    _save_ckpt(name, result)
    print(f"[checkpoint] {name} 저장 완료", flush=True)
    return result


def _fmt_pct(v) -> str:
    return f"{v:+.2f}%" if v is not None else "-"


def run_scenario(data: pf.PortfolioData, cfg: dict, start: date, end: date, candidate: str, **params) -> dict:
    """후보 하나를 돌려 보고에 필요한 지표를 계산한다."""
    result = pf.simulate_portfolio(data, cfg, start, end, candidate, apply_costs=True, apply_tax=True, **params)
    result_pretax = pf.simulate_portfolio(data, cfg, start, end, candidate, apply_costs=True, apply_tax=False, **params)

    clean_rows = [r for r in result.equity_rows if r["total_krw"] is not None]
    equity_metrics = bt.compute_equity_metrics(clean_rows)
    pretax_metrics = bt.compute_equity_metrics([r for r in result_pretax.equity_rows if r["total_krw"] is not None])
    posttax_a = pf.compute_posttax_a(result, data, cfg, end)
    recovery = pf.compute_longest_recovery_days(clean_rows)
    worst = pf.compute_worst_year_month(clean_rows)
    yearly_pct = bt.compute_yearly_returns_from_equity(clean_rows)
    total_tax_krw = round(sum(t["tax_krw"] for t in result.broker.tax_log) + (posttax_a.get("tax_paid_krw") or 0))

    return {
        "candidate": candidate, "result": result, "equity_rows": clean_rows,
        "pretax_cagr_pct": pretax_metrics.get("cagr_pct"),
        "posttax_a_pct": posttax_a.get("cagr_liquidated_pct"), "posttax_b_pct": equity_metrics.get("cagr_pct"),
        "mdd_pct": equity_metrics.get("mdd_pct"), "sharpe": equity_metrics.get("sharpe"),
        "recovery": recovery, "worst_year": worst.get("worst_year"), "worst_month": worst.get("worst_month"),
        "total_tax_krw": total_tax_krw, "trade_count": result.trade_count, "yearly_pct": yearly_pct,
    }


def crash_detail(equity_rows: list, start: date, end: date) -> dict:
    """폭락 구간의 최대 손실(구간 시작 직전 고점 대비)과 회복 시점을 구한다."""
    rows = [r for r in equity_rows if r["total_krw"] is not None]
    if not rows:
        return {}
    dates = [r["date"] for r in rows]
    values = {r["date"]: r["total_krw"] for r in rows}
    before = [d for d in dates if d < start.isoformat()]
    peak_before = max((values[d] for d in before), default=values[dates[0]])
    peak_before = max(peak_before, *(values[d] for d in before if d >= before[0])) if before else values[dates[0]]
    in_window = [d for d in dates if start.isoformat() <= d <= end.isoformat()]
    if not in_window:
        return {}
    trough = min(values[d] for d in in_window)
    max_loss_pct = round((trough / peak_before - 1) * 100, 2) if peak_before else None
    after = [d for d in dates if d > end.isoformat()]
    recovery_date = next((d for d in after if values[d] >= peak_before), None)
    return {"max_loss_pct": max_loss_pct, "recovery_date": recovery_date, "peak_before": round(peak_before)}


def judge_tier(scenario: dict, qqq_scenario: dict, tier: str) -> dict:
    excess_a_pp = round((scenario["posttax_a_pct"] or 0) - (qqq_scenario["posttax_a_pct"] or 0), 2)
    mdd_ok = abs(scenario["mdd_pct"] or 0) <= abs(qqq_scenario["mdd_pct"] or 0)
    if tier == "공격형":
        met = excess_a_pp >= 2.0 and mdd_ok
        return {"tier": tier, "excess_a_pp": excess_a_pp, "mdd_ok": mdd_ok, "pass": met}
    within = abs(excess_a_pp) <= 1.5 or excess_a_pp > 1.5  # -1.5%p 이내(더 좋으면 당연 통과)
    within = excess_a_pp >= -1.5
    mdd_ok_stable = abs(scenario["mdd_pct"] or 0) <= abs(qqq_scenario["mdd_pct"] or 0) * 0.7
    recovery_ok = (scenario["recovery"].get("longest_recovery_trading_days") or 10**9) <= (
        qqq_scenario["recovery"].get("longest_recovery_trading_days") or 0
    )
    met = within and mdd_ok_stable and recovery_ok
    return {"tier": tier, "excess_a_pp": excess_a_pp, "within_1_5pp": within, "mdd_ok_70pct": mdd_ok_stable, "recovery_ok": recovery_ok, "pass": met}


def main() -> Path:
    cfg = bt.load_config()
    with open(PREREG_PATH, encoding="utf-8") as f:
        prereg = yaml.safe_load(f)
    target_tier = prereg.get("target_tier", "undecided")

    print("[p6-1] 데이터 준비(QQQ·QQQM·QLD·DTB3·환율, 1999-03~2021-12) ...", flush=True)
    data = _run_checkpointed("data", lambda: pf.prepare_data(cfg, FULL_START, FULL_END))
    qqq_bench_data = replace(data, core_df=data.qqq_df, core_dividends=data.qqq_dividends)

    periods = {"full": (FULL_START, FULL_END), **SUB_PERIODS}
    candidates = ["QQQ", "P0", "P3", "P4", "P5"]

    scenarios: dict[str, dict] = {}
    for pkey, (start, end) in periods.items():
        for cand in candidates:
            key = f"{cand}_{pkey}"
            d = qqq_bench_data if cand == "QQQ" else data
            run_cand = "P0" if cand == "QQQ" else cand
            print(f"[p6-1] {pkey} 구간 {cand} ...", flush=True)
            scenarios[key] = _run_checkpointed(key, lambda d=d, start=start, end=end, run_cand=run_cand: run_scenario(d, cfg, start, end, run_cand))

    print("[p6-1] 재현 확인(P0 2015~2021 vs P5-5 QQQ 값) ...", flush=True)
    repro = _run_checkpointed("repro_qqq_2015_2021", lambda: run_scenario(qqq_bench_data, cfg, REPRO_START, REPRO_END, "P0"))
    repro_diff = {
        "posttax_a_pct": round((repro["posttax_a_pct"] or 0) - P5_5_QQQ_REFERENCE["posttax_a_pct"], 2),
        "posttax_b_pct": round((repro["posttax_b_pct"] or 0) - P5_5_QQQ_REFERENCE["posttax_b_pct"], 2),
        "mdd_pct": round((repro["mdd_pct"] or 0) - P5_5_QQQ_REFERENCE["mdd_pct"], 2),
    }

    print("[p6-1] 주변값(P3·P5 낙폭 단계) ...", flush=True)
    neighbor_drawdown: dict[str, dict] = {}
    for cand in ("P3", "P5"):
        for triggers in NEIGHBOR_DRAWDOWN_TRIGGERS:
            key = f"{cand}_full_dd{triggers}"
            neighbor_drawdown[key] = _run_checkpointed(
                key, lambda cand=cand, triggers=triggers: run_scenario(data, cfg, FULL_START, FULL_END, cand, drawdown_triggers_pct=triggers)
            )

    print("[p6-1] 주변값(P4·P5 이동평균) ...", flush=True)
    neighbor_sma: dict[str, dict] = {}
    for cand in ("P4", "P5"):
        for sma in NEIGHBOR_SMA_DAYS:
            key = f"{cand}_full_sma{sma}"
            neighbor_sma[key] = _run_checkpointed(key, lambda cand=cand, sma=sma: run_scenario(data, cfg, FULL_START, FULL_END, cand, sma_days=sma))

    print("[p6-1] 폭락 구간 상세 ...", flush=True)
    crash_table: dict[str, dict] = {}
    for cand in candidates:
        full_rows = scenarios[f"{cand}_full"]["equity_rows"]
        crash_table[cand] = {label: crash_detail(full_rows, s, e) for label, (s, e) in CRASH_WINDOWS.items()}

    print("[p6-1] 판정 ...", flush=True)
    tiers_to_judge = ["공격형", "안정형"] if target_tier == "undecided" else [target_tier]
    verdicts: dict[str, dict] = {}
    for cand in ("P0", "P3", "P4", "P5"):
        full_scn = scenarios[f"{cand}_full"]
        qqq_full = scenarios["QQQ_full"]
        sub_hits = 0
        for pkey in SUB_PERIODS:
            sub_scn, qqq_sub = scenarios[f"{cand}_{pkey}"], scenarios[f"QQQ_{pkey}"]
            sub_judge = judge_tier(sub_scn, qqq_sub, tiers_to_judge[0])
            if sub_judge["pass"]:
                sub_hits += 1
        per_tier = {t: judge_tier(full_scn, qqq_full, t) for t in tiers_to_judge}
        common_ok = sub_hits >= 2
        final = "불합격"
        for t in tiers_to_judge:
            if per_tier[t]["pass"] and common_ok:
                final = t
                break
        verdicts[cand] = {"per_tier": per_tier, "sub_hits": sub_hits, "common_ok": common_ok, "final": final}

    run_id = f"{date.today().isoformat()}_{bt.make_run_id(cfg, FULL_START, FULL_END)[-8:]}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    write_report(
        out_dir, cfg, data, scenarios, repro, repro_diff, neighbor_drawdown, neighbor_sma,
        crash_table, verdicts, target_tier, periods,
    )
    print(f"[p6-1] 결과: {out_dir}", flush=True)
    return out_dir


def write_report(
    out_dir: Path, cfg: dict, data: pf.PortfolioData, scenarios: dict, repro: dict, repro_diff: dict,
    neighbor_drawdown: dict, neighbor_sma: dict, crash_table: dict, verdicts: dict, target_tier: str, periods: dict,
) -> None:
    try:
        commit_hash = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit_hash = None

    header = "| 후보 | 세전 | 세후(a) | 세후(b) | QQQ대비(a) | MDD | 원금회복(거래일) | 최악의 해 | 샤프 | 세금합계 | 매매횟수 |"
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"

    def row(cand: str, pkey: str) -> str:
        s = scenarios[f"{cand}_{pkey}"]
        qqq = scenarios[f"QQQ_{pkey}"]
        a_rel = (s["posttax_a_pct"] or 0) - (qqq["posttax_a_pct"] or 0)
        rec = s["recovery"]
        rec_str = f"{rec.get('longest_recovery_trading_days')}{'(진행중)' if rec.get('open_ended') else ''}" if rec else "-"
        wy = s["worst_year"]
        wy_str = f"{wy[0]}: {wy[1]:+.1f}%" if wy else "-"
        return (
            f"| {cand} | {_fmt_pct(s['pretax_cagr_pct'])} | {_fmt_pct(s['posttax_a_pct'])} | {_fmt_pct(s['posttax_b_pct'])} | "
            f"{a_rel:+.2f}%p | {s['mdd_pct']}% | {rec_str} | {wy_str} | {s['sharpe']} | {s['total_tax_krw']:,}원 | {s['trade_count']} |"
        )

    lines = [
        "# P6-1 실험 결과 — 사전 등록 커밋 확인용", "",
        "- 사전 등록: configs/p6_1_preregistration.yaml (커밋 전에 후보·판정 기준 등록, 결과 확인 후 미수정)",
        f"- 목표 등급: {target_tier}",
        f"- 이 실행의 코드 커밋 해시: {commit_hash}", "",
        "## 1장: 데이터", "",
        f"- QLD 합성 검증(2006-06~ 겹치는 기간): {data.qld_synthesis_check}",
        f"- 환율 보완(FRED DEXKOUS): {data.fx_fallback_stats}",
        f"- DTB3 정렬: {data.dtb3_stats}", "",
        "## 재현 확인 (P0=QQQM100%, 2015~2021 vs P5-5 QQQ 값)", "",
        f"- 세후(a) 차이: {repro_diff['posttax_a_pct']:+.2f}%p · 세후(b) 차이: {repro_diff['posttax_b_pct']:+.2f}%p · MDD 차이: {repro_diff['mdd_pct']:+.2f}%p",
        "- 차이가 0에 가까우면 재현 성공(잔여 차이는 QQQM·QQQ 보수율 0.05%p 차이로 설명 가능한 수준이어야 한다).", "",
        "## 2장: 결과 표", "",
    ]
    period_titles = {
        "full": "### 1999-03~2021-12 (전 기간)", "sub_1999_2007": "### 1999-03~2007-12 (닷컴 붕괴 포함)",
        "sub_2008_2015": "### 2008-01~2015-12 (금융위기·회복 포함)", "sub_2016_2021": "### 2016-01~2021-12",
    }
    for pkey in periods:
        lines += [period_titles[pkey], "", header, sep]
        for cand in ("QQQ", "P0", "P3", "P4", "P5"):
            lines.append(row(cand, pkey))
        lines.append("")

    lines += ["### 연도별 수익률 (세후(a), 전 기간, P0·P3·P4·P5)", ""]
    for cand in ("P0", "P3", "P4", "P5"):
        yearly = scenarios[f"{cand}_full"]["yearly_pct"]
        line = " · ".join(f"{y}: {v:+.1f}%" if v is not None else f"{y}: -" for y, v in sorted(yearly.items()))
        lines += [f"- {cand}: {line}", ""]

    lines += ["## 주변값 (전 기간, 세후(a))", "", "### P3·P5 낙폭 투입 단계", ""]
    for cand in ("P3", "P5"):
        main_a = scenarios[f"{cand}_full"]["posttax_a_pct"]
        for triggers in NEIGHBOR_DRAWDOWN_TRIGGERS:
            s = neighbor_drawdown[f"{cand}_full_dd{triggers}"]
            mark = "(주 설정)" if triggers == MAIN_DRAWDOWN_TRIGGERS else ""
            lines.append(f"- {cand} {triggers}{mark}: 세후(a) {_fmt_pct(s['posttax_a_pct'])} (주 설정 대비 {(s['posttax_a_pct'] or 0) - (main_a or 0):+.2f}%p)")
    lines += ["", "### P4·P5 이동평균 일수", ""]
    for cand in ("P4", "P5"):
        main_a = scenarios[f"{cand}_full"]["posttax_a_pct"]
        for sma in NEIGHBOR_SMA_DAYS:
            s = neighbor_sma[f"{cand}_full_sma{sma}"]
            mark = "(주 설정)" if sma == MAIN_SMA_DAYS else ""
            lines.append(f"- {cand} {sma}일{mark}: 세후(a) {_fmt_pct(s['posttax_a_pct'])} (주 설정 대비 {(s['posttax_a_pct'] or 0) - (main_a or 0):+.2f}%p)")

    lines += ["", "## 3장: 판정 (전 기간 기준, 하위 구간 3개 중 2개 이상 충족 = 공통 조건)", ""]
    for cand, v in verdicts.items():
        detail = " · ".join(f"{t}: {'충족' if r['pass'] else '미충족'}({r})" for t, r in v["per_tier"].items())
        lines.append(f"- {cand}: 하위구간 충족 {v['sub_hits']}/3 · {detail} · **최종 판정: {v['final']}**")

    lines += ["", "## 4장: 폭락 구간 상세 (전 기간 시뮬레이션 기준)", ""]
    for cand, windows in crash_table.items():
        lines.append(f"### {cand}")
        for label, d in windows.items():
            if d:
                lines.append(f"- {label}: 최대손실 {d.get('max_loss_pct')}% (직전 고점 {d.get('peak_before'):,}원 대비), 회복일: {d.get('recovery_date') or '구간 끝까지 미회복'}")
        lines.append("")

    lines += ["상태: (완료 보고에 기록)"]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
