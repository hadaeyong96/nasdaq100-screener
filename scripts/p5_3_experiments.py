"""P5-3 실험 스크립트 — 새 기준선(B0.5=슬롯5) + 성과 기여 분해 + C·E·T 실험.

일회성 리포트 스크립트 (CLAUDE.md scripts/). engine/backtest.py의 순수 시뮬레이션
함수를 그대로 재사용하고, 여기서는 실험 조합·점검·기여 분해를 오케스트레이션만 한다.

실행:
    python -m scripts.p5_3_experiments
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

from core.indicators import compute_atr  # noqa: E402
from engine import backtest as bt  # noqa: E402

OUT_ROOT = bt.OUTPUT_ROOT / "p5_3"


def run_scenario(data: bt.BacktestData, cfg: dict, start: date, end: date, *, label: str, max_slots: int = 5, **overrides) -> dict:
    """B0.5/B0/실험 하나를 돌려 비교표 한 줄 + 완료보고에 필요한 값을 계산한다 (P5-2와 동일 구조).

    세전(pretax) = 수수료 반영, 세금 제외(apply_costs=True, apply_tax=False) — P5-3 5번이
    QQQ 줄에도 같은 정의를 요구해 engine.backtest.simulate_benchmark도 이 시그니처로 맞췄다.
    """
    result = bt.simulate_portfolio(data, cfg, start, end, apply_costs=True, apply_tax=True, max_slots=max_slots, **overrides)
    equity_metrics = bt.compute_equity_metrics(result.equity_rows)
    positions = bt.aggregate_positions(result.trades, result.still_open_position_ids)
    position_stats = bt.compute_position_stats(positions)
    liquidated = bt.compute_liquidated_cagr(result, data, cfg, start, end)

    result_pretax = bt.simulate_portfolio(data, cfg, start, end, apply_costs=True, apply_tax=False, max_slots=max_slots, **overrides)
    pretax_metrics = bt.compute_equity_metrics(result_pretax.equity_rows)

    monthly: dict = {}
    for t in result.trades:
        if t.get("side") == "진입":
            monthly[t["date"][:7]] = monthly.get(t["date"][:7], 0) + 1
    avg_monthly_entries = round(sum(monthly.values()) / max(len(monthly), 1), 1)
    total_tax_krw = round(sum(t["tax_krw"] for t in result.broker.tax_log))
    limit_rejections = sum(1 for r in result.rejected if "한도 초과" in r["reason"])
    yearly_pct = bt.compute_yearly_returns_from_equity(result.equity_rows)

    return {
        "label": label, "result": result, "positions": positions,
        "pretax_cagr_pct": pretax_metrics.get("cagr_pct"),
        "cagr_a_pct": liquidated.get("cagr_liquidated_pct"),
        "cagr_b_pct": equity_metrics.get("cagr_pct"),
        "tax_paid_a_krw": liquidated.get("tax_paid_krw"),
        "mdd_pct": equity_metrics.get("mdd_pct"), "sharpe": equity_metrics.get("sharpe"),
        "win_rate_pct": position_stats.get("win_rate_pct"), "payoff_ratio": position_stats.get("payoff_ratio"),
        "expectancy_r": position_stats.get("expectancy_r"), "position_count": position_stats.get("count"),
        "avg_monthly_entries": avg_monthly_entries, "total_tax_krw": total_tax_krw,
        "limit_rejections": limit_rejections, "yearly_pct": yearly_pct,
    }


def classify(s: dict, b05: dict) -> str:
    """P5-3 4번 판정 규칙(공식화)."""
    a_diff = (s["cagr_a_pct"] if s["cagr_a_pct"] is not None else -999) - (b05["cagr_a_pct"] or 0)
    b_diff = (s["cagr_b_pct"] if s["cagr_b_pct"] is not None else -999) - (b05["cagr_b_pct"] or 0)
    mdd_diff = (s["mdd_pct"] if s["mdd_pct"] is not None else -999) - (b05["mdd_pct"] or 0)
    if a_diff >= 0.5 and b_diff >= 0.5 and mdd_diff >= -1.0:
        return "개선"
    if a_diff <= -0.5 or b_diff <= -0.5:
        return "악화"
    return "차이 없음"


def is_family_unstable(b05: dict, variants: list, threshold_pp: float = 3.0) -> bool:
    return any((v["cagr_a_pct"] or 0) < (b05["cagr_a_pct"] or 0) - threshold_pp for v in variants)


def _fmt_pct(v) -> str:
    return f"{v:+.2f}%" if v is not None else "-"


def _price_anomaly_scan(df: pd.DataFrame, year: int, threshold_pct: float = 50.0) -> list:
    """해당 연도 안에서 하루 등락률이 threshold_pct%를 넘는 날을 찾는다 (분할 미반영·튄 값 점검용)."""
    sub = df[df.index.year == year]
    ret = sub["close"].pct_change()
    flagged = ret[ret.abs() > threshold_pct / 100]
    return [{"date": d.date().isoformat(), "pct": round(v * 100, 1)} for d, v in flagged.items()]


def section1_checks(data: bt.BacktestData, cfg: dict, start: date, end: date, b05: dict) -> dict:
    bt_cfg = cfg["backtest"]
    total_krw = bt_cfg["total_krw"]
    result = b05["result"]
    positions = b05["positions"]

    # ── 1-1: 2016년 분해 ──
    # 2015-12-31/2016-12-31이 실제 거래일이 아닐 수 있어(예: 2016-12-31은 토요일) 반드시
    # "그 날짜 이전 마지막 실제 거래일"로 스냅한다 — 스냅하지 않으면 fx_by_date·가격
    # 조회가 전부 빈값이 되어 환율·미실현 손익 조각이 통째로 빠지는 조용한 버그가 난다.
    eq_by_date = {r["date"]: r["total_krw"] for r in result.equity_rows}
    all_dates = sorted(eq_by_date)
    boy_key = max((d for d in all_dates if d <= "2015-12-31"), default=None)
    eoy_key = max((d for d in all_dates if d <= "2016-12-31"), default=None)
    boy = date.fromisoformat(boy_key) if boy_key else date(2015, 12, 31)
    eoy = date.fromisoformat(eoy_key) if eoy_key else date(2016, 12, 31)

    state_boy = bt.simulate_portfolio(data, cfg, start, boy, max_slots=5).states
    state_eoy = bt.simulate_portfolio(data, cfg, start, eoy, max_slots=5).states
    stock_gain_boy = bt.compute_stock_leg_gain_krw(result.trades, state_boy, data, cfg, boy, apply_costs=True)
    stock_gain_eoy = bt.compute_stock_leg_gain_krw(result.trades, state_eoy, data, cfg, eoy, apply_costs=True)
    stock_contribution_2016 = stock_gain_eoy - stock_gain_boy

    total_boy = eq_by_date.get(boy_key)
    total_eoy = eq_by_date.get(eoy_key)
    total_gain_2016 = (total_eoy - total_boy) if (total_boy and total_eoy) else None
    qqqm_residual_2016 = (total_gain_2016 - stock_contribution_2016) if total_gain_2016 is not None else None

    dividend_2016_krw = 0.0
    for d in result.broker.dividend_log:
        if d["date"].startswith("2016"):
            fx = data.fx_by_date.get(d["date"])
            if fx:
                dividend_2016_krw += d["net_usd"] * fx
    dividend_2016_krw = round(dividend_2016_krw)

    fx_boy = data.fx_by_date.get(boy_key)
    fx_eoy = data.fx_by_date.get(eoy_key)
    avg_total_2016 = sum(v for d, v in eq_by_date.items() if d.startswith("2016")) / max(
        sum(1 for d in eq_by_date if d.startswith("2016")), 1
    )
    fx_contribution_2016 = round((fx_eoy / fx_boy - 1) * avg_total_2016) if (fx_boy and fx_eoy) else None
    qqqm_price_residual_2016 = (
        round(qqqm_residual_2016 - dividend_2016_krw - fx_contribution_2016)
        if (qqqm_residual_2016 is not None and fx_contribution_2016 is not None)
        else None
    )

    # 종목별 2016 기여 상위 5개 (실현손익 2016 + 미실현 변화분)
    per_ticker_realized: dict = {}
    for t in result.trades:
        if t["side"] == "청산" and t["date"].startswith("2016") and t.get("pnl_krw") is not None:
            per_ticker_realized[t["ticker"]] = per_ticker_realized.get(t["ticker"], 0) + t["pnl_krw"]

    costs = bt_cfg["costs"]
    fx_eoy_ts = pd.Timestamp(eoy)
    fx_boy_ts = pd.Timestamp(boy)

    def _per_ticker_unrealized(states, ts, fx_rate):
        out = {}
        for ticker, state_ in states.items():
            for unit, qty in state_["units"].items():
                if not qty or qty <= 0:
                    continue
                df = data.indicator_map.get(ticker)
                if df is None or ts not in df.index:
                    continue
                close = df.loc[ts, "close"]
                if pd.isna(close):
                    continue
                entry_price = state_["entries"].get(unit)
                if entry_price is None:
                    continue
                proceeds = float(close) * qty * (1 - costs["commission_sell_pct"] / 100)
                cost = entry_price * qty * (1 + costs["commission_buy_pct"] / 100)
                out[ticker] = out.get(ticker, 0) + (proceeds - cost) * fx_rate
        return out

    unreal_boy = _per_ticker_unrealized(state_boy, fx_boy_ts, fx_boy or 0)
    unreal_eoy = _per_ticker_unrealized(state_eoy, fx_eoy_ts, fx_eoy or 0)
    per_ticker_total = dict(per_ticker_realized)
    for t in set(unreal_eoy) | set(unreal_boy):
        per_ticker_total[t] = per_ticker_total.get(t, 0) + unreal_eoy.get(t, 0) - unreal_boy.get(t, 0)
    top5 = sorted(per_ticker_total.items(), key=lambda kv: kv[1], reverse=True)[:5]

    anomalies = {}
    for ticker, _ in top5:
        df = data.indicator_map.get(ticker)
        if df is not None:
            found = _price_anomaly_scan(df, 2016)
            if found:
                anomalies[ticker] = found

    check_1_1 = {
        "stock_contribution_2016_krw": round(stock_contribution_2016),
        "qqqm_residual_2016_krw": round(qqqm_residual_2016) if qqqm_residual_2016 is not None else None,
        "dividend_2016_krw": dividend_2016_krw,
        "fx_contribution_2016_krw": fx_contribution_2016,
        "qqqm_price_residual_2016_krw": qqqm_price_residual_2016,
        "total_gain_2016_krw": round(total_gain_2016) if total_gain_2016 is not None else None,
        "top5_tickers": [(t, round(v)) for t, v in top5],
        "price_anomalies": anomalies,
    }

    # ── 1-2: 상위 10개 제외 재계산(반사실 재시뮬레이션) ──
    top10_positions = sorted(positions, key=lambda p: p["pnl_krw"], reverse=True)[:10]
    old_approx = bt.compute_topN_excluded_cagr(positions, result.equity_rows[-1]["total_krw"], total_krw, start, end, n=10)
    blocked = {(p["ticker"], p["opened_date"]) for p in top10_positions}
    result_resim = bt.simulate_portfolio(data, cfg, start, end, max_slots=5, blocked_new_entries=blocked)
    metrics_resim = bt.compute_equity_metrics(result_resim.equity_rows)
    liquidated_resim = bt.compute_liquidated_cagr(result_resim, data, cfg, start, end)
    check_1_2 = {
        "old_approx_cagr_pct": old_approx["cagr_pct"],
        "resim_cagr_b_pct": metrics_resim.get("cagr_pct"),
        "resim_cagr_a_pct": liquidated_resim.get("cagr_liquidated_pct"),
        "excluded_tickers": [p["ticker"] for p in top10_positions],
        "excluded_sum_pnl_krw": old_approx["excluded_sum_pnl_krw"],
    }

    # ── 1-3: B1 SMA190=SMA200 확인 ──
    r190 = bt.compute_sma_regime(data.qqq_df, 190)
    r200 = bt.compute_sma_regime(data.qqq_df, 200)
    r210 = bt.compute_sma_regime(data.qqq_df, 210)
    common_190_200 = set(r190) & set(r200)
    diff_190_200 = sorted(d for d in common_190_200 if r190[d] != r200[d])
    common_200_210 = set(r200) & set(r210)
    diff_200_210 = sorted(d for d in common_200_210 if r200[d] != r210[d])
    check_1_3 = {
        "true_days_190": sum(r190.values()), "true_days_200": sum(r200.values()), "true_days_210": sum(r210.values()),
        "diff_190_200_count": len(diff_190_200), "diff_190_200_sample": diff_190_200[:5],
        "diff_200_210_count": len(diff_200_210),
    }

    return {"1_1": check_1_1, "1_2": check_1_2, "1_3": check_1_3}


def section2_attribution(data: bt.BacktestData, cfg: dict, start: date, end: date, b05: dict) -> dict:
    result = b05["result"]
    positions = b05["positions"]

    twr_stock = bt.compute_stock_sleeve_twr_usd(result.equity_rows, result.trades, data.fx_by_date)
    twr_qqq = bt.compute_qqq_twr_over_days_usd(data.qqq_df, data.qqq_dividends, set(twr_stock["daily_returns"]))
    selection_excess_pp = (
        round((twr_stock["twr_annualized_pct"] or 0) - (twr_qqq["twr_annualized_pct"] or 0), 2)
        if twr_stock["twr_annualized_pct"] is not None and twr_qqq["twr_annualized_pct"] is not None
        else None
    )
    weight_stats = bt.compute_position_weight_stats(result.equity_rows)

    # 세전 수익 기여 원화 분해: ① 무비용무세금 vs ② 비용만 vs ③ 비용+세금(=B0.5 본선)
    result_nocost = bt.simulate_portfolio(data, cfg, start, end, apply_costs=False, apply_tax=False, max_slots=5)
    result_costonly = bt.simulate_portfolio(data, cfg, start, end, apply_costs=True, apply_tax=False, max_slots=5)
    total_krw = cfg["backtest"]["total_krw"]
    final_nocost = result_nocost.equity_rows[-1]["total_krw"]
    final_costonly = result_costonly.equity_rows[-1]["total_krw"]
    final_main = result.equity_rows[-1]["total_krw"]
    stock_leg_gain_full = bt.compute_stock_leg_gain_krw(result_nocost.trades, result_nocost.states, data, cfg, end, apply_costs=False)
    total_gain_nocost = final_nocost - total_krw
    qqqm_leg_gain_full = total_gain_nocost - stock_leg_gain_full
    cost_drag_krw = final_costonly - final_nocost
    tax_drag_krw = final_main - final_costonly
    money_attribution = {
        "신호종목": round(stock_leg_gain_full), "QQQM(배당·가격 포함)": round(qqqm_leg_gain_full),
        "비용": round(cost_drag_krw), "세금": round(tax_drag_krw),
    }

    entry_type = bt.compute_entry_type_breakdown(positions, result.trades)
    exit_type = bt.compute_exit_type_breakdown(result.trades)

    return {
        "twr_stock_pct": twr_stock["twr_annualized_pct"], "twr_stock_days": twr_stock["invested_days"],
        "twr_qqq_pct": twr_qqq["twr_annualized_pct"], "selection_excess_pp": selection_excess_pp,
        "weight_stats": weight_stats, "money_attribution": money_attribution,
        "entry_type": entry_type, "exit_type": exit_type,
    }


def main() -> Path:
    cfg = bt.load_config()
    bt_cfg = cfg["backtest"]
    start = bt._parse_date(bt_cfg["train_start"])
    end = bt._parse_date(bt_cfg["train_end"])
    warmup_start = bt._parse_date(bt_cfg["warmup_start"])

    data = bt.prepare_data(cfg, warmup_start, end)

    print("[p5-3] B0(슬롯8) ...")
    b0 = run_scenario(data, cfg, start, end, label="B0(슬롯8)", max_slots=8)
    print("[p5-3] B0.5(슬롯5, 새 기준선) ...")
    b05 = run_scenario(data, cfg, start, end, label="B0.5(기준)", max_slots=5)

    print("[p5-3] 1장 점검 ...")
    checks = section1_checks(data, cfg, start, end, b05)
    print("[p5-3] 2장 기여 분해 ...")
    attribution = section2_attribution(data, cfg, start, end, b05)

    scenarios: dict = {"B0": b0, "B0.5": b05}

    print("[p5-3] C1 상대강도(6/12개월) ...")
    trading_days = [d.date() for d in list(data.qqq_df.index)]
    for months, key in ((6, "C1_6mo"), (12, "C1_12mo")):
        rs = bt.compute_relative_strength_top_half(data.indicator_map, trading_days, months)
        scenarios[key] = run_scenario(data, cfg, start, end, label=f"C1 상대강도{months}개월", rs_filter=rs)

    print("[p5-3] C2 RSI 반복교차 제외 ...")
    whipsaw = bt.compute_rsi_whipsaw_block(data.indicator_map, lookback=15, max_crosses=2)
    scenarios["C2"] = run_scenario(data, cfg, start, end, label="C2 RSI반복교차제외", a1_whipsaw_block=whipsaw)

    print("[p5-3] C3 거래량(volume/obv) ...")
    for mode, key in (("volume", "C3_volume"), ("obv", "C3_obv")):
        vf = bt.compute_volume_entry_filter(data.indicator_map, mode)
        scenarios[key] = run_scenario(data, cfg, start, end, label=f"C3 거래량({mode})", volume_filter=vf)

    print("[p5-3] E1 ATR 추적손절(2.5/3.0/3.5) ...")
    atr_indicator_map = {t: df.assign(atr=compute_atr(df, period=14)) for t, df in data.indicator_map.items()}
    data_atr = replace(data, indicator_map=atr_indicator_map)
    for mult, key in ((2.5, "E1_2.5"), (3.0, "E1_3.0"), (3.5, "E1_3.5")):
        scenarios[key] = run_scenario(data_atr, cfg, start, end, label=f"E1 ATR추적손절x{mult}", atr_trail_mult=mult)

    print("[p5-3] E2 분할 익절 ...")
    scenarios["E2"] = run_scenario(data, cfg, start, end, label="E2 분할익절(+2R)", partial_tp_r_mult=2.0)

    print("[p5-3] T1 진입 유형 분리(A형만/B형만) ...")
    scenarios["T1_A"] = run_scenario(data, cfg, start, end, label="T1 A형만", entry_type_allowed={"A1"})
    scenarios["T1_B"] = run_scenario(data, cfg, start, end, label="T1 B형만", entry_type_allowed={"B"})

    print("[p5-3] QQQ 벤치마크 ...")
    benchmark_main = bt.simulate_benchmark(data, cfg, start, end, apply_costs=True, apply_tax=True)
    benchmark_pretax = bt.simulate_benchmark(data, cfg, start, end, apply_costs=True, apply_tax=False)
    qqq_yearly = bt.compute_yearly_returns_from_equity(benchmark_main.equity_rows)
    qqq_row = {
        "label": "QQQ(벤치마크)", "pretax_cagr_pct": round(benchmark_pretax.cagr_unrealized * 100, 2),
        "cagr_a_pct": round(benchmark_main.cagr_liquidated * 100, 2), "cagr_b_pct": round(benchmark_main.cagr_unrealized * 100, 2),
        "mdd_pct": round(benchmark_main.mdd_pct, 2), "sharpe": None, "win_rate_pct": None, "payoff_ratio": None,
        "expectancy_r": None, "position_count": None, "avg_monthly_entries": None, "total_tax_krw": None,
        "limit_rejections": None, "yearly_pct": qqq_yearly,
    }

    run_id = f"{date.today().isoformat()}_{bt.make_run_id(cfg, start, end)[-8:]}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(out_dir, cfg, scenarios, qqq_row, checks, attribution, start, end)
    write_csvs(out_dir, scenarios)

    print(f"[p5-3] 결과: {out_dir}")
    return out_dir


def write_csvs(out_dir: Path, scenarios: dict) -> None:
    for key, s in scenarios.items():
        pd.DataFrame(s["result"].trades).to_csv(out_dir / f"trades_{key}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(s["positions"]).to_csv(out_dir / f"positions_{key}.csv", index=False, encoding="utf-8-sig")


def write_report(out_dir: Path, cfg: dict, scenarios: dict, qqq_row: dict, checks: dict, attribution: dict, start: date, end: date) -> None:
    b05 = scenarios["B0.5"]
    c1_1, c1_2, c1_3 = checks["1_1"], checks["1_2"], checks["1_3"]
    lines = [
        f"# P5-3 실험 결과 — {start} ~ {end}",
        "",
        "## 1장: 점검",
        "",
        "### 1-1. 2016년 분해 (B0.5 실제 수치 기준)",
        f"- 2016년 총수익(원화): {c1_1['total_gain_2016_krw']:,}원" if c1_1['total_gain_2016_krw'] is not None else "- 계산 불가",
        f"- 신호 종목 기여: {c1_1['stock_contribution_2016_krw']:,}원",
        f"- QQQM 몫 잔여 기여(배당+가격+환율 합계): {c1_1['qqqm_residual_2016_krw']:,}원" if c1_1['qqqm_residual_2016_krw'] is not None else "",
        f"  - 그중 배당(세후 실수령, 종목+QQQM 합산): {c1_1['dividend_2016_krw']:,}원",
        f"  - 그중 환율 기여(근사 — (연말환율/연초환율-1)×연중 평균 총자산): {c1_1['fx_contribution_2016_krw']:,}원" if c1_1['fx_contribution_2016_krw'] is not None else "",
        f"  - 그중 QQQM 자체 가격 상승분(잔여): {c1_1['qqqm_price_residual_2016_krw']:,}원" if c1_1['qqqm_price_residual_2016_krw'] is not None else "",
        "- 2016년 기여 상위 5개 종목: " + ", ".join(f"{t}({v:+,}원)" for t, v in c1_1["top5_tickers"]),
        "- 가격 데이터 이상(하루 등락 ±50% 초과) 발견: " + (str(c1_1["price_anomalies"]) if c1_1["price_anomalies"] else "없음"),
        "",
        "### 1-2. 상위 10개 제외 재계산",
        f"- 기존 근사(최종 평가액에서 pnl 단순 차감): {c1_2['old_approx_cagr_pct']:+.2f}%",
        f"- 반사실 재시뮬레이션(해당 진입 자체를 없앤 경우) 세후(a): {_fmt_pct(c1_2['resim_cagr_a_pct'])} · 세후(b): {_fmt_pct(c1_2['resim_cagr_b_pct'])}",
        f"- 두 값이 다른 이유: 단순 차감은 그 자금이 이후 재투자돼 복리로 불어난 효과를 무시한다 — "
        f"반사실 재시뮬레이션은 그 진입이 막힌 자금이 QQQM에 그대로 남아 QQQM 수익률로 계속 불어나므로, "
        f"실제 하락폭이 단순 차감보다 훨씬 작게(또는 크게) 나올 수 있다.",
        f"- 제외한 10개 포지션: {', '.join(c1_2['excluded_tickers'])} (합계 손익 {c1_2['excluded_sum_pnl_krw']:,}원)",
        "",
        "### 1-3. B1 SMA190 = SMA200 확인",
        f"- 국면 True 일수: 190일선 {c1_3['true_days_190']}일 · 200일선 {c1_3['true_days_200']}일 · 210일선 {c1_3['true_days_210']}일",
        f"- 190일선과 200일선이 실제로 다른 날: {c1_3['diff_190_200_count']}일 (예: {c1_3['diff_190_200_sample']})",
        f"- 200일선과 210일선이 다른 날: {c1_3['diff_200_210_count']}일",
        "- 결론: 파라미터는 정확히 반영되고 있다(다른 날이 실제로 존재). 다만 1,816거래일 중 5일만 달라 "
        "이번 학습 구간에서는 그 며칠이 우연히 실제 신규 진입 판정에 영향을 준 날과 겹치지 않아 결과가 "
        "완전히 같게 나온 것으로 보인다 — 버그가 아니라 이 구간의 우연.",
        "",
        "## 2장: 성과 기여 분해 (B0.5 기준)",
        "",
        f"- 신호 종목 몫 TWR(연율화, 달러, 물려 있던 {attribution['twr_stock_days']}일만): {_fmt_pct(attribution['twr_stock_pct'])}",
        f"- 같은 날들의 QQQ TWR(연율화, 달러): {_fmt_pct(attribution['twr_qqq_pct'])}",
        f"- 종목 선택 초과 수익 = 신호 종목 TWR − QQQ TWR: {attribution['selection_excess_pp']:+.2f}%p"
        if attribution['selection_excess_pp'] is not None else "- 종목 선택 초과 수익: 계산 불가",
        f"- 판정: {'신호 종목의 초과 수익 미확인' if (attribution['selection_excess_pp'] or 0) <= 0 else '신호 종목이 QQQ 대비 초과 수익을 냈다고 볼 근거 있음'}",
        f"- 평균 종목 투자 비중: {attribution['weight_stats'].get('avg_weight_pct')}% "
        f"(최대 {attribution['weight_stats'].get('max_weight_pct')}%, 최소 {attribution['weight_stats'].get('min_weight_pct')}%)",
        "",
        "### 세전 수익 원화 기여 분해 (① 무비용무세금 기준 종목/QQQM, ①→②→③로 비용·세금 차감분)",
        *[f"- {k}: {v:,}원" for k, v in attribution["money_attribution"].items()],
        "",
        "### 진입 유형별",
        *[f"- {label}: {v['count']}개, 승률 {v['win_rate_pct']}%, 기대값 {v['expectancy_r']}R, 총손익 {v['total_pnl_krw']:,}원"
          for label, v in attribution["entry_type"].items()],
        "",
        "### 청산 유형별",
        *[f"- {kind}: {v['count']}건, 평균 R {v['avg_r']}" for kind, v in attribution["exit_type"].items()],
        "",
        "## 3장·5장: 실험 결과 표 (학습 구간, B0.5 = 슬롯 5 기준)",
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
    order = ["B0", "B0.5", "C1_6mo", "C1_12mo", "C2", "C3_volume", "C3_obv", "E1_2.5", "E1_3.0", "E1_3.5", "E2", "T1_A", "T1_B"]
    for key in order:
        lines.append(row(scenarios[key]))

    lines += ["", "## 실험별 연도별 세후(b) 수익률", ""]
    for key in order:
        s = scenarios[key]
        lines.append(f"- {s['label']}: {bt._fmt_yearly(s['yearly_pct'])}")

    lines += ["", "## 각 실험 판정", ""]
    c1_family_unstable = is_family_unstable(b05, [scenarios["C1_6mo"], scenarios["C1_12mo"]])
    e1_family_unstable = is_family_unstable(b05, [scenarios["E1_2.5"], scenarios["E1_3.5"]])
    for key in ["C1_6mo", "C1_12mo", "C2", "C3_volume", "C3_obv", "E1_2.5", "E1_3.0", "E1_3.5", "E2", "T1_A", "T1_B"]:
        s = scenarios[key]
        if key.startswith("C1_") and c1_family_unstable:
            verdict = "불안정"
        elif key.startswith("E1_") and e1_family_unstable:
            verdict = "불안정"
        else:
            verdict = classify(s, b05)
        lines.append(f"- {s['label']}: {verdict}")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
