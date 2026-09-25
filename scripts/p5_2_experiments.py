"""P5-2 실험 스크립트 — 기준선 보완 + 단일 변수 실험 (A1·B1·B2·B4).

일회성 리포트 스크립트 (CLAUDE.md scripts/). engine/backtest.py의 순수 시뮬레이션
함수(simulate_portfolio, compute_liquidated_cagr, aggregate_positions, ...)를 그대로
재사용한다. 여기서는 실험 조합을 돌리고 비교 표·완료 보고용 자료를 만드는
오케스트레이션만 한다 — 새 신호·상태 전이 로직은 없다.

실행:
    python -m scripts.p5_2_experiments
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from data import backtest_prices as bp  # noqa: E402
from engine import backtest as bt  # noqa: E402

OUT_ROOT = bt.OUTPUT_ROOT / "p5_2"


def fetch_cash_parking(kind: str, cfg: dict, warmup_start: date, end: date, qqq_df_raw: pd.DataFrame):
    """B4 실험용 "쉬는 돈" 보관처 시세를 받는다 (네트워크/캐시 I/O — 순수 함수 아님).

    kind: "short_treasury"(config의 b4_short_treasury_ticker, 기본 BIL) | "cash"(이자 0%,
    합성 평평 시세 — QQQ의 거래일 달력을 빌려 종가 1.0 고정, 배당 없음).
    출력: (compute_indicators 적용된 price_df, dividends) — BacktestData.cash_etf_df/
    cash_etf_dividends와 같은 모양.
    """
    bt_cfg = cfg["backtest"]
    if kind == "short_treasury":
        ticker = bt_cfg["b4_short_treasury_ticker"]
        df_raw, _ = bp.fetch_history(ticker, warmup_start, end)
        dividends = bp.fetch_dividends(ticker, warmup_start, end)
        return bt.compute_indicators(df_raw, cfg), dividends
    if kind == "cash":
        idx = qqq_df_raw.index
        flat = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0}, index=idx
        )
        return bt.compute_indicators(flat, cfg), pd.Series(dtype=float)
    raise ValueError(f"알 수 없는 cash_parking kind: {kind}")


def run_scenario(
    data: bt.BacktestData, cfg: dict, start: date, end: date, *,
    label: str, max_slots: int | None = None, regime_ok: dict | None = None,
) -> dict:
    """B0/실험 한 개를 돌려 보고 표 한 줄 + 완료 보고에 필요한 값을 전부 계산한다.

    세전(pretax)은 수수료는 반영하고 세금만 뺀 시나리오다(engine.backtest.run의
    ② 비용만 버킷과 같은 뜻) — 실전에서 수수료는 피할 수 없지만 세금 유무로 순수
    신호 전략의 성과를 보기 위함.
    """
    print(f"[p5-2] {label}: 세후(비용+세금) 시뮬레이션...")
    result = bt.simulate_portfolio(
        data, cfg, start, end, apply_costs=True, apply_tax=True, max_slots=max_slots, regime_ok=regime_ok
    )
    equity_metrics = bt.compute_equity_metrics(result.equity_rows)
    positions = bt.aggregate_positions(result.trades, result.still_open_position_ids)
    position_stats = bt.compute_position_stats(positions)
    liquidated = bt.compute_liquidated_cagr(result, data, cfg, start, end)

    print(f"[p5-2] {label}: 세전(비용만) 시뮬레이션...")
    result_pretax = bt.simulate_portfolio(
        data, cfg, start, end, apply_costs=True, apply_tax=False, max_slots=max_slots, regime_ok=regime_ok
    )
    pretax_metrics = bt.compute_equity_metrics(result_pretax.equity_rows)

    monthly: dict[str, int] = {}
    for t in result.trades:
        if t.get("side") == "진입":
            monthly[t["date"][:7]] = monthly.get(t["date"][:7], 0) + 1
    avg_monthly_entries = round(sum(monthly.values()) / max(len(monthly), 1), 1)
    total_tax_krw = round(sum(t["tax_krw"] for t in result.broker.tax_log))
    limit_rejections = sum(1 for r in result.rejected if "한도 초과" in r["reason"])
    yearly_pct = bt.compute_yearly_returns_from_equity(result.equity_rows)

    return {
        "label": label,
        "result": result,
        "positions": positions,
        "pretax_cagr_pct": pretax_metrics.get("cagr_pct"),
        "cagr_a_pct": liquidated.get("cagr_liquidated_pct"),
        "cagr_b_pct": equity_metrics.get("cagr_pct"),
        "tax_paid_a_krw": liquidated.get("tax_paid_krw"),
        "mdd_pct": equity_metrics.get("mdd_pct"),
        "sharpe": equity_metrics.get("sharpe"),
        "win_rate_pct": position_stats.get("win_rate_pct"),
        "payoff_ratio": position_stats.get("payoff_ratio"),
        "expectancy_r": position_stats.get("expectancy_r"),
        "position_count": position_stats.get("count"),
        "avg_monthly_entries": avg_monthly_entries,
        "total_tax_krw": total_tax_krw,
        "limit_rejections": limit_rejections,
        "yearly_pct": yearly_pct,
    }


def _compound_cagr(yearly_pct: dict) -> float | None:
    """연도별 수익률(%) 사전에서 기하평균 연복리를 근사한다 (거래일 수 가중치 없이 연 단위)."""
    vals = [v for v in yearly_pct.values() if v is not None]
    if not vals:
        return None
    factor = 1.0
    for v in vals:
        factor *= 1 + v / 100
    return round((factor ** (1 / len(vals)) - 1) * 100, 2)


def classify(name: str, s: dict, b0: dict) -> str:
    """P5-2 3장 판정 규칙: 세후 (a)·(b)와 MDD를 함께 본다. 임계값(0.5%p CAGR, 1%p MDD)은
    "판단이 필요한 부분"에 적어 둔 자의적 기준 — 완료 보고에서 사용자에게 확인받는다."""
    a_diff = (s["cagr_a_pct"] or -999) - (b0["cagr_a_pct"] or 0)
    b_diff = (s["cagr_b_pct"] or -999) - (b0["cagr_b_pct"] or 0)
    mdd_diff = (s["mdd_pct"] or -999) - (b0["mdd_pct"] or 0)  # 음수가 더 나쁨(더 크게 마이너스)
    if a_diff > 0.5 and b_diff > 0.5 and mdd_diff >= -1.0:
        return "개선"
    if a_diff < -0.5 and b_diff < -0.5:
        return "악화"
    return "차이 없음"


def is_family_unstable(b0: dict, variants: dict[str, dict], collapse_threshold_pp: float = 15.0) -> bool:
    """주변값 중 하나라도 B0의 세후(a) CAGR보다 collapse_threshold_pp(%p) 넘게 무너지면 불안정."""
    for s in variants.values():
        if (s["cagr_a_pct"] or 0) < (b0["cagr_a_pct"] or 0) - collapse_threshold_pp:
            return True
    return False


def main() -> Path:
    cfg = bt.load_config()
    bt_cfg = cfg["backtest"]
    start = bt._parse_date(bt_cfg["train_start"])
    end = bt._parse_date(bt_cfg["train_end"])
    warmup_start = bt._parse_date(bt_cfg["warmup_start"])

    data = bt.prepare_data(cfg, warmup_start, end)

    scenarios: dict[str, dict] = {}
    scenarios["B0"] = run_scenario(data, cfg, start, end, label="B0(기준)")

    scenarios["A1_slot5"] = run_scenario(data, cfg, start, end, label="A1 슬롯5", max_slots=5)
    scenarios["A1_slot4"] = run_scenario(data, cfg, start, end, label="A1 슬롯4", max_slots=4)

    for window in (190, 200, 210):
        regime = bt.compute_sma_regime(data.qqq_df, window)
        scenarios[f"B1_sma{window}"] = run_scenario(data, cfg, start, end, label=f"B1 SMA{window}", regime_ok=regime)

    cloud_regime = bt.compute_cloud_regime(data.qqq_df)
    scenarios["B2_cloud"] = run_scenario(data, cfg, start, end, label="B2 구름", regime_ok=cloud_regime)

    print("[p5-2] B4: 단기 국채(BIL) 시세를 받는 중...")
    qqq_df_raw, _ = bp.fetch_history(bt_cfg["benchmark_ticker"], warmup_start, end)
    for kind, key in (("short_treasury", "B4_treasury"), ("cash", "B4_cash")):
        price_df, dividends = fetch_cash_parking(kind, cfg, warmup_start, end, qqq_df_raw)
        data_b4 = replace(data, cash_etf_df=price_df, cash_etf_dividends=dividends)
        scenarios[key] = run_scenario(data_b4, cfg, start, end, label=f"B4 {kind}")

    print("[p5-2] QQQ 벤치마크...")
    benchmark = bt.simulate_benchmark(data, cfg, start, end)
    pretax_yearly = bt.compute_pretax_benchmark_yearly_returns(data.qqq_df, data.qqq_dividends, data.fx_by_date, start, end)
    qqq_yearly = bt.compute_yearly_returns_from_equity(benchmark.equity_rows)
    qqq_row = {
        "label": "QQQ(벤치마크)",
        "pretax_cagr_pct": _compound_cagr(pretax_yearly.get("krw_total_return", {})),
        "cagr_a_pct": round(benchmark.cagr_liquidated * 100, 2),
        "cagr_b_pct": round(benchmark.cagr_unrealized * 100, 2),
        "mdd_pct": round(benchmark.mdd_pct, 2),
        "sharpe": None, "win_rate_pct": None, "payoff_ratio": None, "expectancy_r": None,
        "position_count": None, "avg_monthly_entries": None, "total_tax_krw": None, "limit_rejections": None,
        "yearly_pct": qqq_yearly,
    }

    # ── 0-3: -5.88R / 최대 손실 포지션 조사 ──
    b0_positions = scenarios["B0"]["positions"]
    worst = min(b0_positions, key=lambda p: p["r"])
    worst_trades = [t for t in scenarios["B0"]["result"].trades if t.get("position_id") == worst["position_id"]]

    # ── 0-4: 연도별 초과 수익 + 상위10 제외 CAGR ──
    b0_yearly = scenarios["B0"]["yearly_pct"]
    years = sorted(set(b0_yearly) | set(qqq_yearly))
    excess_by_year = {y: round((b0_yearly.get(y) or 0) - (qqq_yearly.get(y) or 0), 2) for y in years}
    final_total_krw = scenarios["B0"]["result"].equity_rows[-1]["total_krw"]
    top10 = bt.compute_topN_excluded_cagr(b0_positions, final_total_krw, bt_cfg["total_krw"], start, end, n=10)

    # ── 국면별 성과 (QQQ 200일선) ──
    regime200 = bt.compute_sma_regime(data.qqq_df, 200)
    regime_split = bt.compute_regime_split_stats(scenarios["B0"]["result"].equity_rows, b0_positions, regime200)

    run_id = f"{date.today().isoformat()}_{bt.make_run_id(cfg, start, end)[-8:]}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(out_dir, cfg, scenarios, qqq_row, worst, worst_trades, excess_by_year, top10, regime_split, start, end)
    write_csvs(out_dir, scenarios, worst_trades)

    print(f"[p5-2] 결과: {out_dir}")
    return out_dir


def _fmt_pct(v) -> str:
    return f"{v:+.2f}%" if v is not None else "-"


def write_csvs(out_dir: Path, scenarios: dict, worst_trades: list) -> None:
    for key, s in scenarios.items():
        pd.DataFrame(s["result"].trades).to_csv(out_dir / f"trades_{key}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(s["positions"]).to_csv(out_dir / f"positions_{key}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(worst_trades).to_csv(out_dir / "worst_position_trades.csv", index=False, encoding="utf-8-sig")


def write_report(
    out_dir: Path, cfg: dict, scenarios: dict, qqq_row: dict, worst: dict, worst_trades: list,
    excess_by_year: dict, top10: dict, regime_split: dict, start: date, end: date,
) -> None:
    b0 = scenarios["B0"]
    lines = [
        f"# P5-2 실험 결과 — {start} ~ {end}",
        "",
        "## 0장: 기준선 보완",
        "",
        f"- B0 세후(a, 기간 끝 전량 청산) CAGR: {_fmt_pct(b0['cagr_a_pct'])} (세금 {b0['tax_paid_a_krw']:,}원)"
        if b0.get("tax_paid_a_krw") is not None else f"- B0 세후(a) CAGR: {_fmt_pct(b0['cagr_a_pct'])}",
        f"- B0 세후(b, 미청산) CAGR: {_fmt_pct(b0['cagr_b_pct'])}",
        f"- QQQ 대비(a): {_fmt_pct((b0['cagr_a_pct'] or 0) - (qqq_row['cagr_a_pct'] or 0))}p"
        f" · QQQ 대비(b): {_fmt_pct((b0['cagr_b_pct'] or 0) - (qqq_row['cagr_b_pct'] or 0))}p",
        "",
        f"### 최대 손실 포지션 조사 ({worst['ticker']}, position_id={worst['position_id']}, R={worst['r']:.2f})",
        f"- 개설일: {worst['opened_date']} · 청산일: {worst['closed_date']} · 손익: {worst['pnl_krw']:,.0f}원 · 초기위험: {worst['initial_risk_krw']:,.0f}원",
        "- 거래 내역:",
    ]
    for t in worst_trades:
        lines.append(
            f"  - {t['date']} {t['side']}({t.get('stage')}) qty={t.get('qty')} price={t.get('price')} "
            f"reason={t.get('reason')} pnl_krw={t.get('pnl_krw')}"
        )
    lines += [
        "",
        "### 연도별 초과 수익 (B0 − QQQ, %p)",
        " · ".join(f"{y}: {v:+.1f}%p" for y, v in excess_by_year.items()),
        "",
        f"### 상위 10개 포지션 제외 CAGR (근사): {top10['cagr_pct']:+.2f}% "
        f"(제외 합계 손익 {top10['excluded_sum_pnl_krw']:,}원, 종목: {', '.join(top10['excluded_tickers'])})",
        "",
        "## 국면별 성과 (B0, QQQ 200일선 기준)",
        f"- 위(above): {regime_split['above']['days']}일, 누적수익률 {regime_split['above']['cum_return_pct']:+.2f}%, "
        f"승률 {regime_split['above']['win_rate_pct']}%, 기대값 {regime_split['above']['expectancy_r']}R "
        f"({regime_split['above']['position_count']}개 포지션)",
        f"- 아래(below): {regime_split['below']['days']}일, 누적수익률 {regime_split['below']['cum_return_pct']:+.2f}%, "
        f"승률 {regime_split['below']['win_rate_pct']}%, 기대값 {regime_split['below']['expectancy_r']}R "
        f"({regime_split['below']['position_count']}개 포지션)",
        "",
        "## 1장·2장: 실험 결과 표 (학습 구간)",
        "",
        "| 설계 | 세전 CAGR | 세후(a) | 세후(b) | QQQ대비(a) | QQQ대비(b) | MDD | 샤프 | 승률 | 손익비 | 기대값R | 포지션수 | 월평균신규진입 | 양도세합계 | 한도초과탈락수 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def row(s):
        a_rel = (s["cagr_a_pct"] or 0) - (qqq_row["cagr_a_pct"] or 0)
        b_rel = (s["cagr_b_pct"] or 0) - (qqq_row["cagr_b_pct"] or 0)
        tax = f"{s['total_tax_krw']:,}" if s.get("total_tax_krw") is not None else "-"
        return (
            f"| {s['label']} | {_fmt_pct(s['pretax_cagr_pct'])} | {_fmt_pct(s['cagr_a_pct'])} | {_fmt_pct(s['cagr_b_pct'])} | "
            f"{a_rel:+.2f}%p | {b_rel:+.2f}%p | {s['mdd_pct']}% | {s['sharpe']} | {s['win_rate_pct']}% | "
            f"{s['payoff_ratio']} | {s['expectancy_r']} | {s['position_count']} | {s['avg_monthly_entries']} | {tax}원 | {s['limit_rejections']} |"
        )

    lines.append(row(qqq_row))
    for key in ("B0", "A1_slot5", "A1_slot4", "B1_sma190", "B1_sma200", "B1_sma210", "B2_cloud", "B4_treasury", "B4_cash"):
        lines.append(row(scenarios[key]))

    lines += ["", "## 실험별 연도별 세후(b) 수익률", ""]
    for key in ("B0", "A1_slot5", "A1_slot4", "B1_sma190", "B1_sma200", "B1_sma210", "B2_cloud", "B4_treasury", "B4_cash"):
        s = scenarios[key]
        lines.append(f"- {s['label']}: {bt._fmt_yearly(s['yearly_pct'])}")

    lines += ["", "## 각 실험 판정", ""]
    b1_family = {k: scenarios[k] for k in ("B1_sma190", "B1_sma210")}
    b1_unstable = is_family_unstable(b0, b1_family)
    for key in ("A1_slot5", "A1_slot4", "B1_sma190", "B1_sma200", "B1_sma210", "B2_cloud", "B4_treasury", "B4_cash"):
        s = scenarios[key]
        verdict = "불안정" if key.startswith("B1_") and b1_unstable else classify(key, s, b0)
        lines.append(f"- {s['label']}: {verdict}")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
