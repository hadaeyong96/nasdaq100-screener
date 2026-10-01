"""P6-3(로드맵 C2) 계좌·납입 구조 연구 실행 (docs/design/p6_3_plan.md, configs/
p6_3_preregistration.yaml — 사전 등록 그대로 실행, 결과 보고 값 보고 설정을 바꾸지 않는다).

engine.portfolio.prepare_data(P6-1과 같은 함수)만 가져다 쓰고 수정하지 않는다. 1999-03~
2021-12 데이터를 한 번만 받아 154개 시작월(1999-03~2011-12) 창마다 잘라 쓴다.

실행:
    python -u -m scripts.p6_3_experiments
    python -u -m scripts.p6_3_experiments --quick   # 처음 3개 창만(검증용)
"""

from __future__ import annotations

import statistics
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import account_sim as sim  # noqa: E402
from engine import portfolio as pf  # noqa: E402
from store import trials as trials_store  # noqa: E402

PREREG_PATH = ROOT / "configs" / "p6_3_preregistration.yaml"
CONFIG_PATH = ROOT / "config.yaml"
OUT_PATH = ROOT / "docs" / "results" / "p6_3.md"
LOG_PATH = ROOT / "outputs" / "p6_3_report.log"

FULL_START = date(1999, 3, 1)


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_prereg() -> dict:
    return yaml.safe_load(PREREG_PATH.read_text(encoding="utf-8"))


def load_main_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _series_to_date_dict(series: pd.Series) -> dict[date, float]:
    if series is None or not len(series):
        return {}
    return {ts.date(): float(v) for ts, v in series.items() if pd.notna(v)}


def _df_close_to_date_dict(df: pd.DataFrame) -> dict[date, float]:
    return {ts.date(): float(v) for ts, v in df["close"].items() if pd.notna(v)}


def month_starts(from_ym: str, to_ym: str) -> list[tuple[int, int]]:
    """"YYYY-MM" ~ "YYYY-MM" 사이 (year, month) 목록(포함, 오름차순)."""
    fy, fm = (int(x) for x in from_ym.split("-"))
    ty, tm = (int(x) for x in to_ym.split("-"))
    out = []
    y, m = fy, fm
    while (y, m) <= (ty, tm):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def first_trading_day_on_or_after(trading_days: list[date], year: int, month: int) -> date | None:
    for d in trading_days:
        if d.year == year and d.month == month:
            return d
    return None


def window_end_month(year: int, month: int, horizon_years: int) -> tuple[int, int]:
    return year + horizon_years, month


CANDIDATE_RUNS = [
    # (결과 라벨, candidate, cost_basis, pension_credit_rate_pct)
    ("K0", "K0", "moving_average", None),
    ("K1_avg", "K1", "moving_average", None),
    ("K1_fifo", "K1", "fifo", None),
    ("K2", "K2", "moving_average", None),
    ("K2b_avg", "K2b", "moving_average", None),
    ("K2b_fifo", "K2b", "fifo", None),
    ("K3_165", "K3", "moving_average", 16.5),
    ("K3_132", "K3", "moving_average", 13.2),
    ("K4", "K4", "moving_average", None),
]


def run_window(
    window_start: date, window_end: date, trading_days: list[date],
    qqq_close: dict, core_close: dict, qqq_div: dict, core_div: dict, fx_by_date: dict, reserve_rate: dict,
    prereg: dict, start_capital_krw: float, saving_krw: float,
) -> dict[str, "sim.CandidateResult"]:
    steps = sim.build_month_steps(
        trading_days, qqq_close, core_close, qqq_div, core_div, fx_by_date, reserve_rate,
        window_start, window_end, prereg["expense_ratio"]["domestic_etf_pct"],
    )
    window_trading_days = [d for d in trading_days if window_start <= d <= window_end]
    events = sim.compute_drawdown_events(window_trading_days, qqq_close, prereg["reserve"]["drawdown_trigger_pct"])
    month_dates = [s.contribution_date for s in steps[:-1]]
    phases = sim.skimming_phase_by_month(month_dates, events) if month_dates else []

    out: dict[str, sim.CandidateResult] = {}
    for label, candidate, cost_basis, credit_rate in CANDIDATE_RUNS:
        out[label] = sim.run_candidate(
            candidate, steps, start_capital_krw, saving_krw, prereg,
            cost_basis=cost_basis, pension_credit_rate_pct=credit_rate,
            skimming_active_by_month=phases if candidate == "K4" else None,
        )
    return out


def summarize(values: list[float]) -> dict:
    if not values:
        return {"median": None, "p10": None, "p90": None, "n": 0}
    s = sorted(values)
    n = len(s)

    def pct(p):
        idx = min(n - 1, max(0, round(p * (n - 1))))
        return s[idx]

    return {"median": statistics.median(s), "p10": pct(0.10), "p90": pct(0.90), "n": n}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="검증용 — 처음 3개 시작월만")
    parser.add_argument("--note", default="", help="trials.db에 남길 추가 메모(예: 버그 수정 후 재실행 사유)")
    args = parser.parse_args()

    prereg = load_prereg()
    main_cfg = load_main_config()
    sim_cfg = prereg["simulation"]
    seal_end = date.fromisoformat(sim_cfg["seal_end"])

    full_end = date(2021, 12, 31)
    sim.enforce_window_not_sealed(full_end, seal_end)

    log("engine.portfolio.prepare_data로 1999-03~2021-12 데이터 준비 중 (한 번만)...")
    data = pf.prepare_data(main_cfg, FULL_START, full_end)
    log(f"거래일 {len(data.core_df.index)}개, QLD·SMA 등은 P6-3에서 쓰지 않음")

    trading_days = sorted(ts.date() for ts in data.core_df.index)
    qqq_close = _df_close_to_date_dict(data.qqq_df)
    core_close = _df_close_to_date_dict(data.core_df)
    qqq_div = _series_to_date_dict(data.qqq_dividends)
    core_div = _series_to_date_dict(data.core_dividends)
    fx_by_date = {date.fromisoformat(k): v for k, v in data.fx_by_date.items() if v is not None}
    reserve_rate = _series_to_date_dict(data.reserve_daily_rate)

    starts = month_starts(sim_cfg["window_start_months"]["from"], sim_cfg["window_start_months"]["to"])
    if args.quick:
        starts = starts[:3]
    log(f"시작월 {len(starts)}개 (quick={args.quick})")

    horizon_years = sim_cfg["horizon_years"]
    start_capital_krw = sim_cfg["start_capital_krw"]
    m_values = [sim_cfg["monthly_saving_krw"]["base"]] + sim_cfg["monthly_saving_krw"]["compare"]

    # results[m][label] = list of (window_start_ym, metric_dict)
    results: dict[int, dict[str, list[dict]]] = {m: {label: [] for label, *_ in CANDIDATE_RUNS} for m in m_values}

    skipped = []
    for wi, (y, mo) in enumerate(starts):
        w_start = first_trading_day_on_or_after(trading_days, y, mo)
        if w_start is None:
            skipped.append(f"{y}-{mo:02d}")
            continue
        ey, emo = window_end_month(y, mo, horizon_years)
        w_end = first_trading_day_on_or_after(trading_days, ey, emo)
        if w_end is None:
            skipped.append(f"{y}-{mo:02d} (끝 {ey}-{emo:02d} 없음)")
            continue
        sim.enforce_window_not_sealed(w_end, seal_end)

        for m in m_values:
            res = run_window(
                w_start, w_end, trading_days, qqq_close, core_close, qqq_div, core_div, fx_by_date, reserve_rate,
                prereg, start_capital_krw, m,
            )
            for label, cr in res.items():
                results[m][label].append({"start": f"{y}-{mo:02d}", "result": cr})
        if (wi + 1) % 20 == 0 or wi == len(starts) - 1:
            log(f"  {wi + 1}/{len(starts)}개 창 완료 (최근: {y}-{mo:02d})")

    if skipped:
        log(f"건너뛴 창 {len(skipped)}개: {skipped}")

    report_lines = [
        "# P6-3(로드맵 C2) 계좌·납입 구조 연구 결과",
        "",
        f"> 실행: `scripts/p6_3_experiments.py` · 사전 등록: `configs/p6_3_preregistration.yaml`"
        f"(해시 비교 없음 — 커밋 `{prereg.get('head_commit_at_registration')}` 이후 변경 금지)",
        f"> 시작월 {len(starts) - len(skipped)}개(건너뜀 {len(skipped)}개) · {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    recommend_candidates = ["K0", "K1_avg", "K1_fifo", "K2", "K2b_avg", "K2b_fifo", "K4"]
    for m in m_values:
        report_lines.append(f"## M = {m:,}원/월")
        report_lines.append("")
        report_lines.append("| 후보 | 연금제외 세후(중앙값) | 하위10% | 상위10% | 1억원 도달(개월, 중앙값) | 세금(중앙값) | 매매횟수(중앙값) |")
        report_lines.append("|---|---|---|---|---|---|---|")
        k0_p10 = None
        medians: dict[str, float] = {}
        for label, *_ in CANDIDATE_RUNS:
            rows = results[m][label]
            vals = [r["result"].final_posttax_ex_pension_krw for r in rows]
            months = [r["result"].months_to_100m_krw for r in rows if r["result"].months_to_100m_krw is not None]
            taxes = [r["result"].total_tax_krw for r in rows]
            trades = [r["result"].trade_count for r in rows]
            s = summarize(vals)
            medians[label] = s["median"]
            if label == "K0":
                k0_p10 = s["p10"]
            month_med = summarize([float(x) for x in months])["median"] if months else None
            report_lines.append(
                f"| {label} | {s['median']:,.0f} | {s['p10']:,.0f} | {s['p90']:,.0f} | "
                f"{month_med if month_med is None else f'{month_med:.0f}'} | {summarize(taxes)['median']:,.0f} | "
                f"{summarize(trades)['median']:.0f} |"
            )
        report_lines.append("")

        # 판정
        candidates_for_rank = [c for c in recommend_candidates if medians.get(c) is not None]
        ranked = sorted(candidates_for_rank, key=lambda c: -medians[c])
        top = ranked[0] if ranked else None
        top_p10 = summarize([r["result"].final_posttax_ex_pension_krw for r in results[m][top]])["p10"] if top else None
        verdict = "추천 보류(하위10%가 K0보다 낮음)"
        if top and k0_p10 is not None and top_p10 is not None and top_p10 >= k0_p10:
            if top == "K0":
                verdict = "K0와 차이 없음/추천할 대안 없음"
            else:
                diff_pct = (medians[top] / medians["K0"] - 1) * 100 if medians.get("K0") else None
                threshold = prereg["verdict"]["difference_threshold_pct"]
                if diff_pct is not None and abs(diff_pct) < threshold:
                    verdict = f"{top}가 중앙값 1위지만 K0와 차이 {diff_pct:.1f}%<{threshold}% — 차이 없음, 관리 쉬운 쪽 권장"
                else:
                    verdict = f"**{top} 추천** (중앙값 {medians[top]:,.0f}원, K0 대비 {diff_pct:+.1f}%)"
        report_lines.append(f"**판정(구조 선택)**: {verdict}")
        report_lines.append("")

        # K3 별도 보고
        for lbl in ("K3_165", "K3_132"):
            rows = results[m][lbl]
            ex_vals = [r["result"].final_posttax_ex_pension_krw for r in rows]
            pa = [r["result"].final_pension_posttax_pension_rate_krw for r in rows]
            pb = [r["result"].final_pension_posttax_lump_sum_krw for r in rows]
            credits = [r["result"].total_tax_credit_krw for r in rows]
            s_ex, s_a, s_b, s_cr = summarize(ex_vals), summarize(pa), summarize(pb), summarize(credits)
            report_lines.append(
                f"- **{lbl}** (별도 보고): 연금제외 세후 중앙값 {s_ex['median']:,.0f}원, "
                f"연금 포함(a:연금수령5.5%) {s_ex['median'] + s_a['median']:,.0f}원, "
                f"연금 포함(b:중도인출16.5%) {s_ex['median'] + s_b['median']:,.0f}원, "
                f"세액공제 합계 중앙값 {s_cr['median']:,.0f}원"
            )
        report_lines.append("")

    # 1999-03 시작 구간 별도 표시
    report_lines.append("## 1999-03 시작 구간(닷컴 붕괴를 바로 맞는 구간)")
    report_lines.append("")
    base_m = sim_cfg["monthly_saving_krw"]["base"]
    for label, *_ in CANDIDATE_RUNS:
        row = next((r for r in results[base_m][label] if r["start"] == "1999-03"), None)
        if row:
            cr = row["result"]
            report_lines.append(f"- {label}: 연금제외 세후 {cr.final_posttax_ex_pension_krw:,.0f}원, 세금 {cr.total_tax_krw:,.0f}원, 매매 {cr.trade_count}회")
    report_lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    log(f"결과 저장: {OUT_PATH}")

    conn = trials_store.connect()
    try:
        trials_store.record_trial(
            conn, run_at=datetime.now().isoformat(timespec="seconds"), label="P6-3 계좌 구조 실행",
            metrics={"windows": len(starts) - len(skipped), "skipped": skipped, "m_values": m_values},
            date_range_start=FULL_START.isoformat(), date_range_end=full_end.isoformat(),
            notes=f"configs/p6_3_preregistration.yaml 그대로 실행, quick={args.quick}" + (f" — {args.note}" if args.note else ""),
        )
    finally:
        conn.close()
    log("store/trials.db에 시도 기록 완료")


if __name__ == "__main__":
    main()
