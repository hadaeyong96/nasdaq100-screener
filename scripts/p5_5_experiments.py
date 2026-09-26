"""P5-5 실험 스크립트 — 과거 구간 확장(2007~2014)으로 B0.5 표본 외 검증 + 2019년 분해.

일회성 리포트 스크립트 (CLAUDE.md scripts/). engine/backtest.py의 순수 시뮬레이션
함수를 그대로 재사용하고, 여기서는 기간 확장·판정·실력검증·분해를 오케스트레이션만
한다. 판정 기준은 configs/p5_5_preregistration.yaml에 이미 커밋되어 있고(결과를
보기 전), 이번 단계는 어떤 전략 파라미터도 바꾸거나 새로 고르지 않는다 — B0.5(P5-3·
P5-4와 완전히 동일)와 K1(참고용, P5-4와 동일 정의)만 더 긴 기간에 재실행한다.

실행:
    python -u -m scripts.p5_5_experiments
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import subprocess
import sys
from dataclasses import replace
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

from core import tax  # noqa: E402
from data import fx as fxmod  # noqa: E402
from data import universe_history as uh  # noqa: E402
from engine import backtest as bt  # noqa: E402

OUT_ROOT = bt.OUTPUT_ROOT / "p5_5"
PREREG_PATH = ROOT / "configs" / "p5_5_preregistration.yaml"

# ── 체크포인트 ──────────────────────────────────────────────────────────────
CHECKPOINT_DIR = bt.OUTPUT_ROOT / "p5_5_checkpoint"


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


# ── 기간·설정 (사전 등록과 동일 — configs/p5_5_preregistration.yaml 참고) ──────
WARMUP_START = date(2006, 1, 1)
PRIMARY_START = date(2007, 1, 1)  # 데이터 신뢰도 점검 후 뒤로 밀릴 수 있다(1장)
TRAIN_END = date(2014, 12, 31)
REPRO_START, REPRO_END = date(2015, 1, 1), date(2021, 12, 31)
FULL_END = date(2021, 12, 31)

# 2007-10-01~2009-03-31 / 2009-03-01~2010-12-31 (지시문 2장, 월 단위 근사)
CRISIS_START, CRISIS_END = date(2007, 10, 1), date(2009, 3, 31)
RECOVERY_START, RECOVERY_END = date(2009, 3, 1), date(2010, 12, 31)

# P5-3(651c818 이전 도입)·P5-4(651c818)에서 이미 나온 B0.5 2015~2021 값 — 재현 확인용.
P5_4_REFERENCE_B05_2015_2021 = {"pretax_cagr_pct": 27.57, "cagr_a_pct": 20.38, "cagr_b_pct": 24.73, "mdd_pct": -26.79}


def run_scenario(data: bt.BacktestData, cfg: dict, start: date, end: date, *, label: str, max_slots: int = 5, **overrides) -> dict:
    """B0.5/K1 하나를 돌려 P5-3 형식 보고에 필요한 값을 계산한다(P5-3·P5-4와 동일 구조)."""
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
        "label": label, "result": result, "result_pretax": result_pretax, "positions": positions,
        "pretax_cagr_pct": pretax_metrics.get("cagr_pct"),
        "cagr_a_pct": liquidated.get("cagr_liquidated_pct"), "cagr_b_pct": equity_metrics.get("cagr_pct"),
        "tax_paid_a_krw": liquidated.get("tax_paid_krw"), "mdd_pct": equity_metrics.get("mdd_pct"),
        "sharpe": equity_metrics.get("sharpe"), "win_rate_pct": position_stats.get("win_rate_pct"),
        "payoff_ratio": position_stats.get("payoff_ratio"), "expectancy_r": position_stats.get("expectancy_r"),
        "position_count": position_stats.get("count"), "avg_monthly_entries": avg_monthly_entries,
        "total_tax_krw": total_tax_krw, "limit_rejections": limit_rejections, "yearly_pct": yearly_pct,
    }


def qqq_row_for(data: bt.BacktestData, cfg: dict, start: date, end: date) -> dict:
    benchmark_main = bt.simulate_benchmark(data, cfg, start, end, apply_costs=True, apply_tax=True)
    benchmark_pretax = bt.simulate_benchmark(data, cfg, start, end, apply_costs=True, apply_tax=False)
    qqq_yearly = bt.compute_yearly_returns_from_equity(benchmark_main.equity_rows)
    return {
        "label": "QQQ(벤치마크)", "result": benchmark_main,
        "pretax_cagr_pct": round(benchmark_pretax.cagr_unrealized * 100, 2),
        "cagr_a_pct": round(benchmark_main.cagr_liquidated * 100, 2), "cagr_b_pct": round(benchmark_main.cagr_unrealized * 100, 2),
        "mdd_pct": round(benchmark_main.mdd_pct, 2), "sharpe": None, "win_rate_pct": None, "payoff_ratio": None,
        "expectancy_r": None, "position_count": None, "avg_monthly_entries": None, "total_tax_krw": None,
        "limit_rejections": None, "yearly_pct": qqq_yearly,
    }


def _fmt_pct(v) -> str:
    return f"{v:+.2f}%" if v is not None else "-"


# ── 1장: 데이터 확장 ─────────────────────────────────────────────────────────


def _first_reliable_year(unreliable_years: set[int], from_year: int, to_year: int) -> int | None:
    """from_year부터 하나씩 뒤로 미루며 신뢰도 낮은 연도가 아닌 첫 해를 찾는다.
    to_year를 넘도록 못 찾으면 None(그 구간 안에는 신뢰 가능한 해가 없음)."""
    y = from_year
    while y in unreliable_years:
        y += 1
        if y > to_year:
            return None
    return y


def section1_data_quality(data: bt.BacktestData, fx_stats: dict) -> dict:
    trading_days = list(data.qqq_df.index)
    coverage = bt.compute_yearly_universe_coverage(data.checkpoints, data.indicator_map, trading_days)
    unreliable_years = sorted(y for y, v in coverage.items() if (v["missing_ratio_pct"] or 0) > 30 and y >= 2007)
    unreliable_set = set(unreliable_years)

    # 표본 외(2007~2014) 전용 — 이 구간 안에 신뢰 가능한 해가 없으면 None(그 판정은 못 한다).
    oos_start_year = _first_reliable_year(unreliable_set, PRIMARY_START.year, TRAIN_END.year)
    # "전체" 구간용 — 2014를 넘어서도 찾는다(있으면 그 해부터라도 전체 구간을 계산한다).
    full_start_year = _first_reliable_year(unreliable_set, PRIMARY_START.year, FULL_END.year)

    return {
        "coverage": coverage, "unreliable_years": unreliable_years, "fx_stats": fx_stats,
        "oos_available": oos_start_year is not None,
        "oos_start": date(oos_start_year, 1, 1) if oos_start_year is not None else None,
        "full_start": date(full_start_year, 1, 1) if full_start_year is not None else None,
    }


def write_missing_tickers_csv(out_dir: Path, coverage: dict) -> Path:
    rows = []
    for year, v in sorted(coverage.items()):
        for ticker in v["missing_members"]:
            rows.append({"year": year, "ticker": ticker})
    path = out_dir / "missing_tickers_2007.csv"
    pd.DataFrame(rows, columns=["year", "ticker"]).to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ── 2장: 국면별 성과 ─────────────────────────────────────────────────────────


def _window_regime(trading_days: list, start: date, end: date) -> dict[str, bool]:
    return {d.date().isoformat(): (start <= d.date() <= end) for d in trading_days}


def section2_regimes(data: bt.BacktestData, b05_full: dict) -> dict:
    result = b05_full["result"]
    positions = b05_full["positions"]
    trading_days = list(data.qqq_df.index)

    sma200 = bt.compute_sma_regime(data.qqq_df, 200)
    above_below = bt.compute_regime_split_stats(result.equity_rows, positions, sma200)

    crisis = bt.compute_regime_split_stats(result.equity_rows, positions, _window_regime(trading_days, CRISIS_START, CRISIS_END))["above"]
    recovery = bt.compute_regime_split_stats(result.equity_rows, positions, _window_regime(trading_days, RECOVERY_START, RECOVERY_END))["above"]

    return {"sma200_above": above_below["above"], "sma200_below": above_below["below"], "crisis_2008": crisis, "recovery": recovery}


# ── 2장: 성과 기여 분해 (2007~2014, P5-3 방식) ────────────────────────────────


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

    total_krw = cfg["backtest"]["total_krw"]

    def _decompose(nocost_key: str, costonly_key: str):
        result_nocost = _run_checkpointed(nocost_key, lambda: bt.simulate_portfolio(data, cfg, start, end, apply_costs=False, apply_tax=False, max_slots=5))
        result_costonly = _run_checkpointed(costonly_key, lambda: bt.simulate_portfolio(data, cfg, start, end, apply_costs=True, apply_tax=False, max_slots=5))
        final_nocost = result_nocost.equity_rows[-1]["total_krw"]
        final_costonly = result_costonly.equity_rows[-1]["total_krw"]
        final_main = result.equity_rows[-1]["total_krw"]
        stock_leg_gain_full = bt.compute_stock_leg_gain_krw(result_nocost.trades, result_nocost.states, data, cfg, end, apply_costs=False)
        total_gain_nocost = final_nocost - total_krw
        qqqm_leg_gain_full = total_gain_nocost - stock_leg_gain_full
        cost_drag_krw = final_costonly - final_nocost
        tax_drag_krw = final_main - final_costonly
        return {"신호종목": round(stock_leg_gain_full), "QQQM(배당·가격 포함)": round(qqqm_leg_gain_full), "비용": round(cost_drag_krw), "세금": round(tax_drag_krw)}

    money_attribution = _decompose("b05_2007_2014_nocost", "b05_2007_2014_costonly")

    entry_type = bt.compute_entry_type_breakdown(positions, result.trades)
    exit_type = bt.compute_exit_type_breakdown(result.trades)

    return {
        "twr_stock_pct": twr_stock["twr_annualized_pct"], "twr_stock_days": twr_stock["invested_days"],
        "twr_qqq_pct": twr_qqq["twr_annualized_pct"], "selection_excess_pp": selection_excess_pp,
        "money_attribution": money_attribution, "entry_type": entry_type, "exit_type": exit_type,
    }


# ── 4장: 2019년 분해 (2015~2021 B0.5 기준) ────────────────────────────────────


def section4_2019_decomposition(data: bt.BacktestData, cfg: dict, b05_repro: dict) -> dict:
    result = b05_repro["result"]
    total_krw = cfg["backtest"]["total_krw"]

    eq_by_date = {r["date"]: r["total_krw"] for r in result.equity_rows}
    all_dates = sorted(eq_by_date)
    boy_key = max((d for d in all_dates if d <= "2018-12-31"), default=None)
    eoy_key = max((d for d in all_dates if d <= "2019-12-31"), default=None)
    boy = date.fromisoformat(boy_key) if boy_key else date(2018, 12, 31)
    eoy = date.fromisoformat(eoy_key) if eoy_key else date(2019, 12, 31)

    state_boy = bt.simulate_portfolio(data, cfg, REPRO_START, boy, max_slots=5).states
    state_eoy = bt.simulate_portfolio(data, cfg, REPRO_START, eoy, max_slots=5).states
    stock_gain_boy = bt.compute_stock_leg_gain_krw(result.trades, state_boy, data, cfg, boy, apply_costs=True)
    stock_gain_eoy = bt.compute_stock_leg_gain_krw(result.trades, state_eoy, data, cfg, eoy, apply_costs=True)
    stock_contribution_2019 = stock_gain_eoy - stock_gain_boy

    total_boy, total_eoy = eq_by_date.get(boy_key), eq_by_date.get(eoy_key)
    total_gain_2019 = (total_eoy - total_boy) if (total_boy and total_eoy) else None
    qqqm_residual_2019 = (total_gain_2019 - stock_contribution_2019) if total_gain_2019 is not None else None

    dividend_2019_krw = 0.0
    for d in result.broker.dividend_log:
        if d["date"].startswith("2019"):
            fx_rate = data.fx_by_date.get(d["date"])
            if fx_rate:
                dividend_2019_krw += d["net_usd"] * fx_rate
    dividend_2019_krw = round(dividend_2019_krw)

    fx_boy, fx_eoy = data.fx_by_date.get(boy_key), data.fx_by_date.get(eoy_key)
    avg_total_2019 = sum(v for d, v in eq_by_date.items() if d.startswith("2019")) / max(sum(1 for d in eq_by_date if d.startswith("2019")), 1)
    fx_contribution_2019 = round((fx_eoy / fx_boy - 1) * avg_total_2019) if (fx_boy and fx_eoy) else None
    qqqm_price_residual_2019 = (
        round(qqqm_residual_2019 - dividend_2019_krw - fx_contribution_2019)
        if (qqqm_residual_2019 is not None and fx_contribution_2019 is not None) else None
    )

    costs = cfg["backtest"]["costs"]
    fx_eoy_ts, fx_boy_ts = pd.Timestamp(eoy), pd.Timestamp(boy)

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

    per_ticker_realized: dict = {}
    for t in result.trades:
        if t["side"] == "청산" and t["date"].startswith("2019") and t.get("pnl_krw") is not None:
            per_ticker_realized[t["ticker"]] = per_ticker_realized.get(t["ticker"], 0) + t["pnl_krw"]
    unreal_boy = _per_ticker_unrealized(state_boy, fx_boy_ts, fx_boy or 0)
    unreal_eoy = _per_ticker_unrealized(state_eoy, fx_eoy_ts, fx_eoy or 0)
    per_ticker_total = dict(per_ticker_realized)
    for t in set(unreal_eoy) | set(unreal_boy):
        per_ticker_total[t] = per_ticker_total.get(t, 0) + unreal_eoy.get(t, 0) - unreal_boy.get(t, 0)
    ranked = sorted(per_ticker_total.items(), key=lambda kv: kv[1], reverse=True)
    top5, bottom5 = ranked[:5], ranked[-5:]

    qqqm_trades_2019 = result.broker.qqqm_trade_count_by_year.get(2019, 0)
    gain_2019 = result.broker.realized_gain_by_year.get(2019, 0)
    qqqm_gain_2019 = result.broker.qqqm_realized_gain_by_year.get(2019, 0)
    tax_cfg = cfg["backtest"]["tax"]
    tax_on_total_2019 = tax.capital_gains_tax(gain_2019, tax_cfg["capital_gains_deduction_krw"], tax_cfg["capital_gains_rate_pct"] / 100)
    tax_on_non_qqqm_2019 = tax.capital_gains_tax(gain_2019 - qqqm_gain_2019, tax_cfg["capital_gains_deduction_krw"], tax_cfg["capital_gains_rate_pct"] / 100)
    qqqm_attributable_tax_2019 = round(tax_on_total_2019 - tax_on_non_qqqm_2019)

    weight_stats_2019 = bt.compute_position_weight_stats([r for r in result.equity_rows if r["date"].startswith("2019")])
    qqq_2019_up = bt.compute_yearly_returns_from_equity(
        bt.simulate_benchmark(data, cfg, REPRO_START, eoy, apply_costs=True, apply_tax=True).equity_rows
    ).get(2019)

    conclusion = (
        "신호 종목 몫이 2019년 급등장을 대부분 놓쳤다(투자 비중이 낮고 회전율이 높아 QQQ 상승분을 다 태우지 못함)"
        if (stock_contribution_2019 or 0) < (total_gain_2019 or 0) * 0.5
        else "QQQM 몫보다 신호 종목 몫이 2019년 부진의 주원인"
    )

    return {
        "total_gain_2019_krw": round(total_gain_2019) if total_gain_2019 is not None else None,
        "stock_contribution_2019_krw": round(stock_contribution_2019),
        "qqqm_residual_2019_krw": round(qqqm_residual_2019) if qqqm_residual_2019 is not None else None,
        "dividend_2019_krw": dividend_2019_krw,
        "fx_contribution_2019_krw": fx_contribution_2019,
        "qqqm_price_residual_2019_krw": qqqm_price_residual_2019,
        "top5_tickers": [(t, round(v)) for t, v in top5],
        "bottom5_tickers": [(t, round(v)) for t, v in bottom5],
        "qqqm_trades_2019": qqqm_trades_2019,
        "qqqm_attributable_tax_2019_krw": qqqm_attributable_tax_2019,
        "avg_weight_pct_2019": weight_stats_2019.get("avg_weight_pct"),
        "qqq_2019_pct": qqq_2019_up,
        "conclusion": conclusion,
    }


# ── 몬테카를로(P5-4와 완전히 동일한 방식·시드 규칙) ────────────────────────────


def _mc_single_draw(draw_i: int, seed_base: int, ctx: dict) -> float:
    close_swing_by_ticker = ctx["close_swing_by_ticker"]
    tech_mask_by_ticker = ctx["tech_mask_by_ticker"]
    active_by_date = ctx["active_by_date"]
    fx_by_date = ctx["fx_by_date"]
    commission_buy_pct = ctx["commission_buy_pct"]
    commission_sell_pct = ctx["commission_sell_pct"]

    rng = np.random.default_rng(seed_base + draw_i)
    draw_stock_gain_krw = 0.0
    for t in ctx["real_entries"]:
        entry_date = t["date"]
        usd_amount = t["price"] * t["qty"]
        candidates = active_by_date.get(entry_date, [])
        if not candidates:
            continue
        picked = None
        for _ in range(10):
            cand = candidates[rng.integers(0, len(candidates))]
            df = close_swing_by_ticker[cand]
            ts = pd.Timestamp(entry_date)
            if ts not in df.index:
                continue
            row = df.loc[ts]
            swing_low, close = row.get("swing_low"), row.get("close")
            if pd.isna(swing_low) or pd.isna(close) or close <= 0 or swing_low >= close:
                continue
            picked = (cand, df, ts, float(close), float(swing_low))
            break
        if picked is None:
            continue
        cand, df, ts, close, swing_low = picked
        qty = usd_amount / close
        sub_close = df["close"].loc[df.index > ts]
        if sub_close.empty:
            continue
        stop_mask = sub_close <= swing_low
        tech_mask = tech_mask_by_ticker[cand].loc[tech_mask_by_ticker[cand].index > ts]
        combined = stop_mask | tech_mask
        if combined.any():
            first_ts = combined[combined].index[0]
            exit_price = float(sub_close.loc[first_ts])
            exit_fx = fx_by_date.get(first_ts.date().isoformat())
        else:
            exit_price = float(sub_close.iloc[-1])
            exit_fx = fx_by_date.get(sub_close.index[-1].date().isoformat())
        exit_fx = exit_fx or fx_by_date.get(entry_date)
        cost_usd = usd_amount * (1 + commission_buy_pct / 100)
        proceeds_usd = qty * exit_price * (1 - commission_sell_pct / 100)
        gain_usd = proceeds_usd - cost_usd
        draw_stock_gain_krw += gain_usd * (exit_fx or 0)

    draw_pretax_total_gain = ctx["qqqm_leg_pretax"] + draw_stock_gain_krw
    draw_pretax_final = ctx["total_krw"] + draw_pretax_total_gain
    tax_krw = tax.capital_gains_tax(draw_pretax_total_gain, ctx["deduction_total"], ctx["rate"]) if draw_pretax_total_gain > 0 else 0.0
    draw_final_after_tax = draw_pretax_final - tax_krw
    years = ctx["years"]
    cagr = (draw_final_after_tax / ctx["total_krw"]) ** (1 / years) - 1 if draw_final_after_tax > 0 and years > 0 else -1.0
    return cagr * 100


_MC_CTX: dict | None = None


def _mc_worker_init(ctx: dict) -> None:
    global _MC_CTX
    _MC_CTX = ctx


def _mc_worker(args: tuple[int, int]) -> tuple[int, float]:
    draw_i, seed_base = args
    return draw_i, _mc_single_draw(draw_i, seed_base, _MC_CTX)


def monte_carlo_luck_test(
    data: bt.BacktestData, cfg: dict, scenario: dict, start: date, end: date,
    n_draws: int = 500, seed_base: int = 0, batch_size: int = 50, ckpt_prefix: str = "mc",
) -> dict:
    """P5-4와 완전히 동일한 방식(같은 근사·같은 시드 규칙) — 실제 진입 이벤트는 그대로 두고
    종목만 그날의 활성 나스닥100 중 무작위로 뽑아 같은 청산 규칙으로 시뮬레이션한다."""
    bt_cfg = cfg["backtest"]
    total_krw = bt_cfg["total_krw"]
    costs = bt_cfg["costs"]
    tax_cfg = bt_cfg["tax"]
    years = (end - start).days / 365.25

    result = scenario["result"]
    result_pretax = scenario["result_pretax"]
    real_entries = [t for t in result.trades if t["side"] == "진입" and (t.get("price") or 0) > 0 and (t.get("qty") or 0) > 0]

    real_pretax_final = result_pretax.equity_rows[-1]["total_krw"]
    real_pretax_stock_gain = bt.compute_stock_leg_gain_krw(result_pretax.trades, result_pretax.states, data, cfg, end, apply_costs=True)
    qqqm_leg_pretax = real_pretax_final - total_krw - real_pretax_stock_gain

    active_by_date: dict[str, list] = {}
    for ts in data.qqq_df.index:
        d = ts.date()
        members = uh.universe_on(data.checkpoints, d) if data.checkpoints else set(data.indicator_map.keys())
        active_by_date[d.isoformat()] = [t for t in members if t in data.indicator_map and ts in data.indicator_map[t].index]

    tech_mask_by_ticker = {}
    close_swing_by_ticker = {}
    for ticker, df in data.indicator_map.items():
        e1_hit = df["dc"].fillna(False)
        prev_rsi = df["rsi"].shift(1)
        e2_hit = ((prev_rsi >= 50) & (df["rsi"] < 50)).fillna(False)
        e3_hit = ((df["close"] < df["cloud_bot"]) | df["chikou_broken"].fillna(False)).fillna(False)
        tech_mask_by_ticker[ticker] = (e1_hit | e2_hit | e3_hit)
        close_swing_by_ticker[ticker] = df[["close", "swing_low"]]

    ctx = {
        "close_swing_by_ticker": close_swing_by_ticker, "tech_mask_by_ticker": tech_mask_by_ticker,
        "active_by_date": active_by_date, "fx_by_date": data.fx_by_date, "real_entries": real_entries,
        "deduction_total": tax_cfg["capital_gains_deduction_krw"] * max(round(years), 1),
        "rate": tax_cfg["capital_gains_rate_pct"] / 100,
        "commission_buy_pct": costs["commission_buy_pct"], "commission_sell_pct": costs["commission_sell_pct"],
        "total_krw": total_krw, "years": years, "qqqm_leg_pretax": qqqm_leg_pretax,
    }

    n_workers = max(1, (os.cpu_count() or 2) - 1)
    n_batches = (n_draws + batch_size - 1) // batch_size
    draws_by_i: dict[int, float] = {}
    for b in range(n_batches):
        lo, hi = b * batch_size, min(b * batch_size + batch_size, n_draws)

        def _compute_batch(lo=lo, hi=hi):
            print(f"[mc:{ckpt_prefix}] draw {lo}~{hi - 1} 계산 중 (워커 {n_workers}개) ...", flush=True)
            tasks = [(i, seed_base) for i in range(lo, hi)]
            with mp.Pool(processes=n_workers, initializer=_mc_worker_init, initargs=(ctx,)) as pool:
                return dict(pool.map(_mc_worker, tasks))

        batch = _run_checkpointed(f"{ckpt_prefix}_batch{b:02d}", _compute_batch)
        draws_by_i.update(batch)

    draw_cagr_a = [draws_by_i[i] for i in range(n_draws)]
    real_cagr_a = scenario["cagr_a_pct"]
    percentile = (
        round(sum(1 for v in draw_cagr_a if v <= real_cagr_a) / len(draw_cagr_a) * 100, 1) if draw_cagr_a and real_cagr_a is not None else None
    )
    verdict = "무작위보다 확실히 나음" if (percentile or 0) >= 95 else ("애매" if (percentile or 0) >= 50 else "무작위 수준 이하")
    return {
        "real_cagr_a_pct": real_cagr_a, "percentile": percentile, "verdict": verdict,
        "draw_mean_pct": round(float(np.mean(draw_cagr_a)), 2) if draw_cagr_a else None,
        "draw_p5_pct": round(float(np.percentile(draw_cagr_a, 5)), 2) if draw_cagr_a else None,
        "draw_p95_pct": round(float(np.percentile(draw_cagr_a, 95)), 2) if draw_cagr_a else None,
    }


# ── 3장: B0.5 판정 (사전 등록 기준) ────────────────────────────────────────────


def judge_b05(b05_oos: dict, qqq_oos: dict, mc_oos: dict) -> tuple[str, dict]:
    excess_a_pp = round((b05_oos["cagr_a_pct"] or 0) - (qqq_oos["cagr_a_pct"] or 0), 2)
    mdd_ok = abs(b05_oos["mdd_pct"] or 0) <= abs(qqq_oos["mdd_pct"] or 0)
    mc_ok = (mc_oos["percentile"] or 0) >= 95
    excess_ok = excess_a_pp >= 1.0
    conditions_met = sum([excess_ok, mdd_ok, mc_ok])

    if excess_a_pp < -1.0:
        verdict = "불합격"
    elif excess_ok and mdd_ok and mc_ok:
        verdict = "합격"
    else:
        verdict = "보류"
    return verdict, {"excess_a_pp": excess_a_pp, "excess_ok": excess_ok, "mdd_ok": mdd_ok, "mc_ok": mc_ok, "conditions_met": conditions_met}


# ── main ────────────────────────────────────────────────────────────────────


def main() -> Path:
    cfg = bt.load_config()

    print("[p5-5] 데이터 확장(2006 워밍업 ~ 2021) 준비 중 ...", flush=True)
    data_raw = _run_checkpointed("data", lambda: bt.prepare_data(cfg, WARMUP_START, FULL_END))

    print("[p5-5] FRED 환율 보완 ...", flush=True)
    fx_fred = _run_checkpointed("fx_fred", lambda: fxmod.fetch_fred_series_range("DEXKOUS", WARMUP_START, FULL_END))
    fx_merged, fx_stats = fxmod.merge_fx_with_fallback(data_raw.fx_by_date, fx_fred)
    data = replace(data_raw, fx_by_date=fx_merged)

    print("[p5-5] 1장 데이터 신뢰도(연도별 구성종목·누락 비율) ...", flush=True)
    sec1 = section1_data_quality(data, fx_stats)
    full_start = sec1["full_start"]
    oos_available = sec1["oos_available"]
    oos_start = sec1["oos_start"]
    print(
        f"[p5-5] 표본 외(2007~2014) 신뢰 가능 여부: {oos_available} (오ㅅ 시작 {oos_start}) · "
        f"전체 구간 실제 시작: {full_start} · 신뢰도 낮은 연도: {sec1['unreliable_years']}",
        flush=True,
    )

    periods = {"repro": (REPRO_START, REPRO_END, "학습 구간 재현 확인"), "full": (full_start, FULL_END, "전체")}
    if oos_available:
        periods["oos"] = (oos_start, TRAIN_END, "표본 외(주 대상)")

    whipsaw_15_2 = bt.compute_rsi_whipsaw_block(data.indicator_map, lookback=15, max_crosses=2)

    scenarios: dict = {}
    for key, (start, end, _role) in periods.items():
        print(f"[p5-5] {key} 구간({start}~{end}) B0.5 ...", flush=True)
        scenarios[f"b05_{key}"] = _run_checkpointed(f"b05_{key}", lambda start=start, end=end: run_scenario(data, cfg, start, end, label="B0.5(기준)", max_slots=5))
        print(f"[p5-5] {key} 구간 K1(참고용) ...", flush=True)
        scenarios[f"k1_{key}"] = _run_checkpointed(
            f"k1_{key}", lambda start=start, end=end: run_scenario(data, cfg, start, end, label="K1(B0.5+C2, 참고용)", max_slots=5, a1_whipsaw_block=whipsaw_15_2)
        )
        print(f"[p5-5] {key} 구간 QQQ ...", flush=True)
        scenarios[f"qqq_{key}"] = _run_checkpointed(f"qqq_{key}", lambda start=start, end=end: qqq_row_for(data, cfg, start, end))

    print("[p5-5] 2장 국면별 성과 ...", flush=True)
    sec2_regimes = section2_regimes(data, scenarios["b05_full"])

    sec2_attr = None
    mc_oos = None
    if oos_available:
        print("[p5-5] 2장 성과 기여 분해(표본 외) ...", flush=True)
        sec2_attr = section2_attribution(data, cfg, oos_start, TRAIN_END, scenarios["b05_oos"])
        print("[p5-5] 2장 무작위 진입 비교(표본 외) ...", flush=True)
        mc_oos = monte_carlo_luck_test(data, cfg, scenarios["b05_oos"], oos_start, TRAIN_END, n_draws=500, seed_base=5000, ckpt_prefix="mc_oos")
    else:
        print("[p5-5] 표본 외(2007~2014) 구간에 신뢰 가능한 해가 없어 그 구간 전용 분석(성과 기여 분해·무작위 진입 비교·판정)을 건너뜀", flush=True)

    print("[p5-5] 2장 무작위 진입 비교(전체) ...", flush=True)
    mc_full = monte_carlo_luck_test(data, cfg, scenarios["b05_full"], full_start, FULL_END, n_draws=500, seed_base=6000, ckpt_prefix="mc_full")

    print("[p5-5] 2장 부트스트랩(전체) ...", flush=True)

    def _bootstrap_full():
        eq_by_date = {r["date"]: r["total_krw"] for r in scenarios["b05_full"]["result"].equity_rows}
        qqq_by_date = {r["date"]: r["total_krw"] for r in scenarios["qqq_full"]["result"].equity_rows}
        common = sorted(set(eq_by_date) & set(qqq_by_date))
        daily_excess = []
        for i in range(1, len(common)):
            d0, d1 = common[i - 1], common[i]
            strat_ret = eq_by_date[d1] / eq_by_date[d0] - 1 if eq_by_date[d0] else 0.0
            qqq_ret = qqq_by_date[d1] / qqq_by_date[d0] - 1 if qqq_by_date[d0] else 0.0
            daily_excess.append(strat_ret - qqq_ret)
        return bt.compute_block_bootstrap_ci(daily_excess, block_size=20, n_resamples=2000, seed=20260926)

    bootstrap_full = _run_checkpointed("bootstrap_full", _bootstrap_full)

    print("[p5-5] 4장 2019년 분해 ...", flush=True)
    sec4 = section4_2019_decomposition(data, cfg, scenarios["b05_repro"])

    print("[p5-5] 3장 B0.5 판정 ...", flush=True)
    if oos_available:
        verdict, verdict_detail = judge_b05(scenarios["b05_oos"], scenarios["qqq_oos"], mc_oos)
    else:
        verdict = "판정 불가(데이터 부족)"
        verdict_detail = {
            "reason": (
                f"사전 등록된 판정 구간(2007~2014) 안에 종목·일수 누락 비율 30% 이하인 해가 하나도 없다 "
                f"(1장 표 참고 — 인수합병 등으로 상장폐지된 뒤 yfinance가 시세를 아예 지워버린 종목이 많다). "
                f"판정 기준·기간을 사후에 바꾸지 않기로 사전 등록했으므로, 이 구간에서는 판정을 내리지 않는다."
            )
        }

    print("[p5-5] 2015~2021 재현 확인 ...", flush=True)
    repro_diff = {
        k: round((scenarios["b05_repro"].get(k) or 0) - v, 2) for k, v in P5_4_REFERENCE_B05_2015_2021.items()
    }

    run_id = f"{date.today().isoformat()}_{bt.make_run_id(cfg, full_start, FULL_END)[-8:]}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_missing_tickers_csv(out_dir, sec1["coverage"])

    write_report(
        out_dir, cfg, scenarios, sec1, sec2_regimes, sec2_attr, mc_oos, mc_full, bootstrap_full, sec4,
        verdict, verdict_detail, repro_diff, oos_start, TRAIN_END, full_start,
    )
    print(f"[p5-5] 결과: {out_dir}", flush=True)
    return out_dir


def write_report(
    out_dir: Path, cfg: dict, scenarios: dict, sec1: dict, sec2_regimes: dict, sec2_attr: dict | None,
    mc_oos: dict | None, mc_full: dict, bootstrap_full: dict, sec4: dict, verdict: str, verdict_detail: dict,
    repro_diff: dict, oos_start: date | None, oos_end: date, full_start: date,
) -> None:
    try:
        commit_hash = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit_hash = None

    def table_row(s):
        qqq = scenarios[f"qqq_{s['_period']}"]
        a_rel = (s["cagr_a_pct"] or 0) - (qqq["cagr_a_pct"] or 0)
        b_rel = (s["cagr_b_pct"] or 0) - (qqq["cagr_b_pct"] or 0)
        tax_s = f"{s['total_tax_krw']:,}" if s.get("total_tax_krw") is not None else "-"
        return (
            f"| {s['label']} | {_fmt_pct(s['pretax_cagr_pct'])} | {_fmt_pct(s['cagr_a_pct'])} | {_fmt_pct(s['cagr_b_pct'])} | "
            f"{a_rel:+.2f}%p | {b_rel:+.2f}%p | {s['mdd_pct']}% | {s['sharpe']} | {s['win_rate_pct']}% | "
            f"{s['payoff_ratio']} | {s['expectancy_r']} | {s['position_count']} | {tax_s}원 |"
        )

    header = "| 설계 | 세전 | 세후(a) | 세후(b) | QQQ대비(a) | QQQ대비(b) | MDD | 샤프 | 승률 | 손익비 | 기대값R | 포지션 | 양도세 |"
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"

    lines = [
        f"# P5-5 실험 결과 — 사전 등록 커밋 확인용",
        "",
        f"- 사전 등록: configs/p5_5_preregistration.yaml (커밋 전에 판정 기준 등록, 결과 확인 후 미수정)",
        f"- 이 실행의 engine/backtest.py 커밋 해시: {commit_hash}",
        (
            f"- 표본 외(2007~2014) 실제 시작 연도: {oos_start.year} (사유: 아래 1장 참고)"
            if oos_start is not None
            else "- **표본 외(2007~2014) 구간: 신뢰 가능한 해가 없어 판정 불가** (아래 1장·3장 참고)"
        ),
        f"- '전체' 구간 실제 시작 연도: {full_start.year}",
        "",
        "## 1장: 데이터 확장",
        "",
        "| 연도 | 구성 종목 수 | 시세 받은 종목 수 | 종목·일수 누락 비율 |",
        "| --- | --- | --- | --- |",
    ]
    for y, v in sorted(sec1["coverage"].items()):
        lines.append(f"| {y} | {v['constituent_count']} | {v['priced_count']} | {v['missing_ratio_pct']}% |")
    lines += [
        "",
        f"- 신뢰도 낮은(누락 30% 초과) 연도: {sec1['unreliable_years'] or '없음'}",
        f"- 누락 종목 목록: missing_tickers_2007.csv",
        f"- 환율 보완: yfinance KRW=X 우선 + FRED DEXKOUS로 빈 날짜 보완 — "
        f"겹치는 {sec1['fx_stats']['overlap_days']}일의 평균 절대 차이 {sec1['fx_stats']['mean_abs_diff']}원, "
        f"최대 차이 {sec1['fx_stats']['max_abs_diff']}원({sec1['fx_stats']['max_abs_diff_date']}), "
        f"FRED로 채운 날짜 {sec1['fx_stats']['filled_from_fallback_days']}일",
        "- 세금 규칙: 과거 실제 세법이 아니라 지금 규칙(연 250만원 공제 후 22%)을 전 기간에 똑같이 적용",
        "- 실적 필터: 기존 백테스트와 동일하게 전체 기간 미적용(engine/backtest.py는 earnings_date=None 고정, P5-1 설계 그대로)",
        "",
        "## 2장: 결과 표",
        "",
    ]
    period_titles = [("repro", "### 2015~2021 (학습 구간 재현 확인)"), ("full", f"### {full_start.year}~2021 (전체)")]
    if oos_start is not None:
        period_titles.insert(0, ("oos", f"### {oos_start.year}~{oos_end.year} (표본 외, 주 대상)"))
    for key, title in period_titles:
        lines += [title, "", header, sep]
        for prefix in ("qqq", "b05", "k1"):
            s = dict(scenarios[f"{prefix}_{key}"])
            s["_period"] = key
            lines.append(table_row(s))
        lines.append("")
    if oos_start is None:
        lines += ["### 2007~2014 (표본 외, 주 대상)", "", "표 없음 — 신뢰 가능한(누락 30% 이하) 해가 이 구간에 하나도 없다.", ""]

    lines += [
        f"### 연도별 초과 수익 (세후 (a), B0.5 − QQQ, {full_start.year}~2021)",
        "",
    ]
    b05_yearly = scenarios["b05_full"]["yearly_pct"]
    qqq_yearly = scenarios["qqq_full"]["yearly_pct"]
    excess_line = " · ".join(
        f"{y}: {(b05_yearly.get(y, 0) - qqq_yearly.get(y, 0)):+.1f}%p" for y in sorted(set(b05_yearly) | set(qqq_yearly))
    )
    lines += [excess_line, "", "### 국면별 성과 (B0.5, 세후(b) 누적수익률)", ""]
    for label, key in (("QQQ 200일선 위", "sma200_above"), ("QQQ 200일선 아래", "sma200_below"), ("2008년 금융위기(2007-10~2009-03)", "crisis_2008"), ("회복기(2009-03~2010-12)", "recovery")):
        v = sec2_regimes[key]
        lines.append(f"- {label}: {v['days']}일, 누적수익 {v['cum_return_pct']:+.2f}%, 승률 {v['win_rate_pct']}%, 기대값 {v['expectancy_r']}R, 포지션 {v['position_count']}개")

    lines += ["", "### 무작위 진입 비교 (500회, P5-4와 같은 방식·시드 규칙)"]
    if mc_oos is not None:
        lines.append(
            f"- {oos_start.year}~2014: 실제 세후(a) {_fmt_pct(mc_oos['real_cagr_a_pct'])} — {mc_oos['percentile']}백분위 → **{mc_oos['verdict']}** "
            f"(무작위 평균 {mc_oos['draw_mean_pct']}%, 5~95백분위 [{mc_oos['draw_p5_pct']}%, {mc_oos['draw_p95_pct']}%])"
        )
    else:
        lines.append("- 2007~2014: 표본 외 구간 없음(1장 참고)으로 계산 안 함")
    lines.append(
        f"- {full_start.year}~2021: 실제 세후(a) {_fmt_pct(mc_full['real_cagr_a_pct'])} — {mc_full['percentile']}백분위 → **{mc_full['verdict']}** "
        f"(무작위 평균 {mc_full['draw_mean_pct']}%, 5~95백분위 [{mc_full['draw_p5_pct']}%, {mc_full['draw_p95_pct']}%])"
    )
    lines += [
        "",
        f"### 부트스트랩 (20일 블록, 2,000회, {full_start.year}~2021 연 초과수익 95% 구간, 참고용)",
        f"- [{bootstrap_full.get('ci_low_pct')}%, {bootstrap_full.get('ci_high_pct')}%] — 0 포함 여부: "
        f"{'포함(유의하지 않음)' if bootstrap_full.get('includes_zero') else '미포함(유의)'}",
        "",
    ]
    if sec2_attr is not None:
        lines += [
            f"### 성과 기여 분해 ({oos_start.year}~2014, B0.5)",
            f"- 신호 종목 몫 TWR(연율화, 달러): {_fmt_pct(sec2_attr['twr_stock_pct'])} (물려 있던 {sec2_attr['twr_stock_days']}일)",
            f"- 같은 날들의 QQQ TWR(연율화, 달러): {_fmt_pct(sec2_attr['twr_qqq_pct'])}",
            f"- 종목 선택 초과 수익: {sec2_attr['selection_excess_pp']:+.2f}%p" if sec2_attr['selection_excess_pp'] is not None else "- 종목 선택 초과 수익: 계산 불가",
            "- 세전 수익 원화 기여 분해: " + " · ".join(f"{k} {v:,}원" for k, v in sec2_attr["money_attribution"].items()),
            "- 진입 유형별: " + " · ".join(f"{label} {v['count']}개(기대값 {v['expectancy_r']}R)" for label, v in sec2_attr["entry_type"].items()),
            "- 청산 유형별: " + " · ".join(f"{kind} {v['count']}건(평균 {v['avg_r']}R)" for kind, v in sec2_attr["exit_type"].items()),
        ]
    else:
        lines += ["### 성과 기여 분해 (2007~2014, B0.5)", "표본 외 구간 없음(1장 참고)으로 계산 안 함"]

    lines += [
        "",
        "### 2015~2021 재현 확인 (P5-3·P5-4 기준값과 비교)",
        f"- 세전 CAGR 차이: {repro_diff['pretax_cagr_pct']:+.2f}%p · 세후(a) 차이: {repro_diff['cagr_a_pct']:+.2f}%p · "
        f"세후(b) 차이: {repro_diff['cagr_b_pct']:+.2f}%p · MDD 차이: {repro_diff['mdd_pct']:+.2f}%p",
        "- 차이가 0에 가까우면 재현 성공. 차이가 있으면 원인(데이터 갱신·구성 종목 변경 등)을 판단이 필요한 부분에 적는다.",
        "",
        "## 3장: B0.5 판정",
        "",
    ]
    if "conditions_met" in verdict_detail:
        lines += [
            f"- 조건별: 세후(a) 초과 {verdict_detail['excess_a_pp']:+.2f}%p({'충족' if verdict_detail['excess_ok'] else '미충족'}) · "
            f"MDD {'충족' if verdict_detail['mdd_ok'] else '미충족'} · 무작위 백분위 {'충족' if verdict_detail['mc_ok'] else '미충족'} "
            f"({verdict_detail['conditions_met']}/3)",
        ]
    else:
        lines += [f"- 사유: {verdict_detail.get('reason', '')}"]
    lines += [
        f"- **판정: {verdict}**",
        "",
        "## 4장: 2019년 분해",
        "",
        f"- 2019년 총수익(원화): {sec4['total_gain_2019_krw']:,}원" if sec4['total_gain_2019_krw'] is not None else "- 계산 불가",
        f"- 신호 종목 기여: {sec4['stock_contribution_2019_krw']:,}원",
        f"- QQQM 몫 잔여 기여(배당+가격+환율): {sec4['qqqm_residual_2019_krw']:,}원" if sec4['qqqm_residual_2019_krw'] is not None else "",
        f"  - 배당(세후): {sec4['dividend_2019_krw']:,}원 · 환율 기여(근사): {sec4['fx_contribution_2019_krw']:,}원" if sec4['fx_contribution_2019_krw'] is not None else "",
        f"  - QQQM 가격 상승분(잔여): {sec4['qqqm_price_residual_2019_krw']:,}원" if sec4['qqqm_price_residual_2019_krw'] is not None else "",
        f"- 2019년 상위 5개: {', '.join(f'{t}({v:+,}원)' for t, v in sec4['top5_tickers'])}",
        f"- 2019년 하위 5개: {', '.join(f'{t}({v:+,}원)' for t, v in sec4['bottom5_tickers'])}",
        f"- 2019년 QQQM 매매(스윕) 횟수: {sec4['qqqm_trades_2019']}회 · 그로 인한 양도세(추정): {sec4['qqqm_attributable_tax_2019_krw']:,}원",
        f"- 2019년 평균 종목 투자 비중: {sec4['avg_weight_pct_2019']}% (참고: 2019년 QQQ 세후(b) {_fmt_pct(sec4['qqq_2019_pct'])})",
        f"- 한 줄 결론: {sec4['conclusion']}",
        "",
        "상태: (완료 보고에 기록)",
    ]

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
