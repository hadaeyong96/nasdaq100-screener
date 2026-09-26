"""P5-4 실험 스크립트 — C2 견고성 + E1 배수 확장 + 조합(K1·K2) + 실력/운 검증 + Walk-Forward.

일회성 리포트 스크립트 (CLAUDE.md scripts/). engine/backtest.py의 순수 시뮬레이션 함수를
그대로 재사용하고, 여기서는 격자 탐색·몬테카를로·부트스트랩·워크포워드·최종 후보 고정을
오케스트레이션만 한다.

실행:
    python -m scripts.p5_4_experiments
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
from core.indicators import compute_atr  # noqa: E402
from data import universe_history as uh  # noqa: E402
from engine import backtest as bt  # noqa: E402

OUT_ROOT = bt.OUTPUT_ROOT / "p5_4"
CONFIG_OUT = ROOT / "configs" / "p5_final_candidate.yaml"

# ── 체크포인트(중단 후 재시작 시 끝난 단위는 다시 계산하지 않는다) ──────────────
CHECKPOINT_DIR = bt.OUTPUT_ROOT / "p5_4_checkpoint"


def _ckpt_path(name: str) -> Path:
    return CHECKPOINT_DIR / f"{name}.pkl"


def _load_ckpt(name: str):
    path = _ckpt_path(name)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_ckpt(name: str, obj) -> None:
    """임시 파일에 쓰고 원자적으로 교체한다 — 저장 도중 프로세스가 죽어도 체크포인트가 깨지지 않게."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _ckpt_path(name)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)


def _run_checkpointed(name: str, compute_fn):
    """이름이 같은 체크포인트가 있으면 불러오고, 없으면 compute_fn()을 실행해 저장한다."""
    cached = _load_ckpt(name)
    if cached is not None:
        print(f"[checkpoint] {name} <- 불러옴", flush=True)
        return cached
    print(f"[checkpoint] {name} 계산 중 ...", flush=True)
    result = compute_fn()
    _save_ckpt(name, result)
    print(f"[checkpoint] {name} 저장 완료", flush=True)
    return result


def run_scenario(data: bt.BacktestData, cfg: dict, start: date, end: date, *, label: str, max_slots: int = 5, **overrides) -> dict:
    """B0.5/실험 하나를 돌려 P5-3 형식 보고에 필요한 값을 계산한다(전 단계와 동일 구조).
    result_pretax(비용만·세금없음)까지 반환해 4장 몬테카를로 기준선 계산에 재사용한다."""
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


def run_scenario_light(data: bt.BacktestData, cfg: dict, start: date, end: date, max_slots: int = 5, **overrides) -> dict:
    """격자 탐색·워크포워드 내부용 — (a)·(b)·MDD만 필요할 때 pretax 재실행·포지션 집계를 뺀 경량판."""
    result = bt.simulate_portfolio(data, cfg, start, end, apply_costs=True, apply_tax=True, max_slots=max_slots, **overrides)
    equity_metrics = bt.compute_equity_metrics(result.equity_rows)
    liquidated = bt.compute_liquidated_cagr(result, data, cfg, start, end)
    return {"cagr_a_pct": liquidated.get("cagr_liquidated_pct"), "cagr_b_pct": equity_metrics.get("cagr_pct"), "mdd_pct": equity_metrics.get("mdd_pct"), "result": result}


def _fmt_pct(v) -> str:
    return f"{v:+.2f}%" if v is not None else "-"


# ── 1장: C2 견고성 ────────────────────────────────────────────────────────


def section1_c2_robustness(data: bt.BacktestData, cfg: dict, start: date, end: date, b05: dict) -> dict:
    grid = {}
    for N in (10, 15, 20):
        for K in (2, 3):

            def _compute(N=N, K=K):
                whipsaw = bt.compute_rsi_whipsaw_block(data.indicator_map, lookback=N, max_crosses=K)
                return run_scenario(data, cfg, start, end, label=f"C2 N{N}K{K}", a1_whipsaw_block=whipsaw)

            grid[(N, K)] = _run_checkpointed(f"c2_grid_N{N}_K{K}", _compute)

    extra = _run_checkpointed("c2_extra", lambda: _compute_c2_extra(data, cfg, start, end, b05, grid))
    return {"grid": grid, "resim": extra["resim"], "blocked_signal_r": extra["blocked_signal_r"]}


def _compute_c2_extra(data: bt.BacktestData, cfg: dict, start: date, end: date, b05: dict, grid: dict) -> dict:
    """C2(15일·2회) 상위 10개 제외 반사실 + C2가 걸러낸 A1 신호의 R 분포. c2_extra 체크포인트 1개로 묶는다."""
    c2_15_2 = grid[(15, 2)]
    positions = c2_15_2["positions"]
    total_krw = cfg["backtest"]["total_krw"]
    top10 = sorted(positions, key=lambda p: p["pnl_krw"], reverse=True)[:10]
    blocked = {(p["ticker"], p["opened_date"]) for p in top10}
    whipsaw_15_2 = bt.compute_rsi_whipsaw_block(data.indicator_map, lookback=15, max_crosses=2)
    result_resim = bt.simulate_portfolio(data, cfg, start, end, max_slots=5, a1_whipsaw_block=whipsaw_15_2, blocked_new_entries=blocked)
    metrics_resim = bt.compute_equity_metrics(result_resim.equity_rows)
    liquidated_resim = bt.compute_liquidated_cagr(result_resim, data, cfg, start, end)
    resim = {
        "before_cagr_a_pct": c2_15_2["cagr_a_pct"], "after_cagr_a_pct": liquidated_resim.get("cagr_liquidated_pct"),
        "after_cagr_b_pct": metrics_resim.get("cagr_pct"), "excluded_tickers": [p["ticker"] for p in top10],
        "drop_pp": round((c2_15_2["cagr_a_pct"] or 0) - (liquidated_resim.get("cagr_liquidated_pct") or 0), 2),
    }

    # C2(15,2)가 걸러낸 A1 신호들의 R 분포 — B0.5(무필터)에서 실제로 연 포지션 중,
    # 그 신호일(체결일 전날)이 이 필터에 걸렸을 것들만 골라 R을 모은다.
    trading_days_list = sorted({d.date().isoformat() for d in data.qqq_df.index})
    idx_of = {d: i for i, d in enumerate(trading_days_list)}
    entry_by_pos = {t["position_id"]: t for t in b05["result"].trades if t["side"] == "진입" and t.get("initial_risk_krw") is not None}
    blocked_r = []
    for p in b05["positions"]:
        entry_trade = entry_by_pos.get(p["position_id"])
        if not entry_trade or entry_trade.get("stage") != "A1":
            continue
        i = idx_of.get(p["opened_date"])
        if i is None or i == 0:
            continue
        signal_date = trading_days_list[i - 1]
        if whipsaw_15_2.get((p["ticker"], signal_date), False):
            blocked_r.append(p["r"])

    blocked_stats = {}
    if blocked_r:
        blocked_stats = {
            "count": len(blocked_r), "mean_r": round(float(np.mean(blocked_r)), 2),
            "median_r": round(float(np.median(blocked_r)), 2), "win_rate_pct": round(sum(1 for r in blocked_r if r > 0) / len(blocked_r) * 100, 1),
            "min_r": round(min(blocked_r), 2), "max_r": round(max(blocked_r), 2),
        }

    return {"resim": resim, "blocked_signal_r": blocked_stats}


# ── 2장: E1 배수 확장 ─────────────────────────────────────────────────────


def section2_e1_extended(data_atr: bt.BacktestData, cfg: dict, start: date, end: date) -> dict:
    mults = [3.5, 4.0, 4.5, 5.0]
    grid = {}
    for m in mults:
        grid[m] = _run_checkpointed(f"e1_mult_{m}", lambda m=m: run_scenario(data_atr, cfg, start, end, label=f"E1 x{m}", atr_trail_mult=m))
    metric_by_mult = {m: grid[m]["cagr_a_pct"] for m in mults}
    chosen = bt.select_flat_region_smallest(mults, metric_by_mult, tol_pp=1.0)

    atr_trail_stats = {}
    for m in mults:
        trades = grid[m]["result"].trades
        risk_by_pos = {t["position_id"]: t["initial_risk_krw"] for t in trades if t["side"] == "진입" and t.get("initial_risk_krw")}
        atr_exits = [t for t in trades if t.get("stage") == "ATR_TRAIL"]
        rs = [t["pnl_krw"] / risk_by_pos[t["position_id"]] for t in atr_exits if t.get("pnl_krw") is not None and risk_by_pos.get(t["position_id"])]
        atr_trail_stats[m] = {"count": len(atr_exits), "avg_r": round(float(np.mean(rs)), 2) if rs else None}

    return {"grid": grid, "mults": mults, "chosen_mult": chosen, "atr_trail_stats": atr_trail_stats}


# ── 4장: 실력인지 운인지 ───────────────────────────────────────────────────


def _mc_single_draw(draw_i: int, seed_base: int, ctx: dict) -> float:
    """몬테카를로 1회분 계산 — monte_carlo_luck_test의 draw 루프 본문을 그대로 옮긴 것
    (로직 변경 없음, 멀티프로세싱 워커에서 호출하기 위해 분리만 함)."""
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


_MC_CTX: dict | None = None  # 워커 프로세스별 전역 — 풀 생성 시 한 번만 채워서 매 draw 재전송을 피한다


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
    """실제 진입 이벤트(날짜·투입 USD)는 그대로 두고 종목만 그날의 활성 나스닥100 중
    무작위로 뽑아 같은 청산 규칙(E1·E2·E3·손절, 단일 로트로 단순화 — find_first_technical_exit)
    으로 시뮬레이션한다. 세금은 전체 실현손익에 한 번에 22%(연 공제 누적)를 적용하는
    근사다(연도별 실현 타이밍은 무시) — 결과가 정밀한 (a) 재현이 아니라 "무작위 대비
    분포에서 몇 번째냐"를 보는 상대 비교용이라 이 근사를 허용했다(판단이 필요한 부분).

    draw는 batch_size(기본 50)개씩 묶어 체크포인트하고, 배치 안에서는 멀티프로세싱으로
    병렬 실행한다. 각 draw는 seed_base+draw_i로 시드가 고정되므로 병렬 순서와 무관하게
    항상 같은 결과가 나온다(재현 가능).
    """
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
        lo = b * batch_size
        hi = min(lo + batch_size, n_draws)

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


def bootstrap_and_yearly(data: bt.BacktestData, scenario: dict, benchmark_main) -> dict:
    eq_by_date = {r["date"]: r["total_krw"] for r in scenario["result"].equity_rows}
    qqq_by_date = {r["date"]: r["total_krw"] for r in benchmark_main.equity_rows}
    common_dates = sorted(set(eq_by_date) & set(qqq_by_date))
    daily_excess = []
    for i in range(1, len(common_dates)):
        d0, d1 = common_dates[i - 1], common_dates[i]
        strat_ret = eq_by_date[d1] / eq_by_date[d0] - 1 if eq_by_date[d0] else 0.0
        qqq_ret = qqq_by_date[d1] / qqq_by_date[d0] - 1 if qqq_by_date[d0] else 0.0
        daily_excess.append(strat_ret - qqq_ret)

    boot = bt.compute_block_bootstrap_ci(daily_excess, block_size=20, n_resamples=2000, seed=20260925)

    yearly_strat = bt.compute_yearly_returns_from_equity(scenario["result"].equity_rows)
    yearly_qqq = bt.compute_yearly_returns_from_equity(benchmark_main.equity_rows)
    years = sorted(set(yearly_strat) & set(yearly_qqq))
    excess_by_year = {y: round((yearly_strat[y] or 0) - (yearly_qqq[y] or 0), 2) for y in years}
    positive_years = sum(1 for v in excess_by_year.values() if v > 0)

    return {"bootstrap": boot, "excess_by_year": excess_by_year, "positive_years": positive_years, "total_years": len(years)}


# ── 5장: Walk-Forward ────────────────────────────────────────────────────


def _walk_forward_window(data: bt.BacktestData, data_atr: bt.BacktestData, cfg: dict, qqq_yearly: dict, train_end_year: int, test_year: int) -> dict:
    c2_grid_params = [(N, K) for N in (10, 15, 20) for K in (2, 3)]
    e1_mults = [3.5, 4.0, 4.5, 5.0]
    train_start = date(2015, 1, 1)
    train_end = date(train_end_year, 12, 31)
    test_start = date(test_year, 1, 1)
    test_end = date(test_year, 12, 31)

    c2_scores = {}
    for N, K in c2_grid_params:
        whipsaw = bt.compute_rsi_whipsaw_block(data.indicator_map, lookback=N, max_crosses=K)
        r = run_scenario_light(data, cfg, train_start, train_end, max_slots=5, a1_whipsaw_block=whipsaw)
        c2_scores[(N, K)] = r["cagr_a_pct"] or -999
    best_nk = max(c2_scores, key=c2_scores.get)

    e1_scores = {}
    for m in e1_mults:
        r = run_scenario_light(data_atr, cfg, train_start, train_end, max_slots=5, atr_trail_mult=m)
        e1_scores[m] = r["cagr_a_pct"] or -999
    chosen_mult = bt.select_flat_region_smallest(e1_mults, e1_scores, tol_pp=1.0)

    whipsaw_test = bt.compute_rsi_whipsaw_block(data.indicator_map, lookback=best_nk[0], max_crosses=best_nk[1])
    if chosen_mult is not None:
        test_result = run_scenario_light(data_atr, cfg, test_start, test_end, max_slots=5, a1_whipsaw_block=whipsaw_test, atr_trail_mult=chosen_mult)
    else:
        test_result = run_scenario_light(data, cfg, test_start, test_end, max_slots=5, a1_whipsaw_block=whipsaw_test)

    return {
        "train_end": train_end.isoformat(), "test_year": test_year, "chosen_nk": best_nk, "chosen_mult": chosen_mult,
        "test_cagr_a_pct": test_result["cagr_a_pct"], "test_cagr_b_pct": test_result["cagr_b_pct"],
        "qqq_cagr_b_pct": qqq_yearly.get(test_year), "excess_b_pp": round((test_result["cagr_b_pct"] or 0) - (qqq_yearly.get(test_year) or 0), 2),
    }


def walk_forward(data: bt.BacktestData, data_atr: bt.BacktestData, cfg: dict, qqq_yearly: dict) -> list:
    windows = [(2017, 2018), (2018, 2019), (2019, 2020), (2020, 2021)]
    out = []
    for train_end_year, test_year in windows:
        w = _run_checkpointed(
            f"wf_test{test_year}", lambda tey=train_end_year, ty=test_year: _walk_forward_window(data, data_atr, cfg, qqq_yearly, tey, ty)
        )
        out.append(w)
    return out


def decide_final_candidate(k1_diag: dict, k2_diag: dict | None) -> tuple[str | None, dict]:
    def score(d):
        if d is None:
            return -1, {}
        mc_ok = (d["mc"]["percentile"] or 0) >= 75
        boot_ok = (not d["boot"]["bootstrap"].get("includes_zero", True)) and d["boot"]["bootstrap"].get("point_estimate_pct", 0) > 0
        wf_ok = d["wf_positive_years"] >= 3
        return sum([mc_ok, boot_ok, wf_ok]), {"mc_ok": mc_ok, "boot_ok": boot_ok, "wf_ok": wf_ok}

    s_k1, detail_k1 = score(k1_diag)
    s_k2, detail_k2 = score(k2_diag)
    if s_k2 >= 2 and s_k2 >= s_k1:
        return "K2", detail_k2
    if s_k1 >= 2:
        return "K1", detail_k1
    return None, {"k1": detail_k1, "k2": detail_k2}


def main() -> Path:
    cfg = bt.load_config()
    bt_cfg = cfg["backtest"]
    start = bt._parse_date(bt_cfg["train_start"])
    end = bt._parse_date(bt_cfg["train_end"])
    warmup_start = bt._parse_date(bt_cfg["warmup_start"])

    data = bt.prepare_data(cfg, warmup_start, end)
    atr_indicator_map = {t: df.assign(atr=compute_atr(df, period=14)) for t, df in data.indicator_map.items()}
    data_atr = replace(data, indicator_map=atr_indicator_map)

    print("[p5-4] B0.5(기준) ...", flush=True)
    b05 = _run_checkpointed("b05", lambda: run_scenario(data, cfg, start, end, label="B0.5(기준)", max_slots=5))

    print("[p5-4] 1장 C2 견고성(6칸 격자) ...", flush=True)
    sec1 = section1_c2_robustness(data, cfg, start, end, b05)

    print("[p5-4] 2장 E1 배수 확장(3.5~5.0) ...", flush=True)
    sec2 = section2_e1_extended(data_atr, cfg, start, end)

    print("[p5-4] 3장 조합(K1·K2) ...", flush=True)
    whipsaw_15_2 = bt.compute_rsi_whipsaw_block(data.indicator_map, lookback=15, max_crosses=2)
    k1 = _run_checkpointed("k1", lambda: run_scenario(data, cfg, start, end, label="K1(B0.5+C2)", a1_whipsaw_block=whipsaw_15_2))
    chosen_mult = sec2["chosen_mult"]
    k2 = None
    if chosen_mult is not None:
        k2 = _run_checkpointed(
            "k2",
            lambda: run_scenario(
                data_atr, cfg, start, end, label=f"K2(K1+E1x{chosen_mult})", a1_whipsaw_block=whipsaw_15_2, atr_trail_mult=chosen_mult
            ),
        )

    print("[p5-4] QQQ 벤치마크 ...", flush=True)
    benchmark_main = _run_checkpointed("benchmark_main", lambda: bt.simulate_benchmark(data, cfg, start, end, apply_costs=True, apply_tax=True))
    benchmark_pretax = _run_checkpointed(
        "benchmark_pretax", lambda: bt.simulate_benchmark(data, cfg, start, end, apply_costs=True, apply_tax=False)
    )
    qqq_yearly = bt.compute_yearly_returns_from_equity(benchmark_main.equity_rows)
    qqq_row = {
        "label": "QQQ(벤치마크)", "pretax_cagr_pct": round(benchmark_pretax.cagr_unrealized * 100, 2),
        "cagr_a_pct": round(benchmark_main.cagr_liquidated * 100, 2), "cagr_b_pct": round(benchmark_main.cagr_unrealized * 100, 2),
        "mdd_pct": round(benchmark_main.mdd_pct, 2), "sharpe": None, "win_rate_pct": None, "payoff_ratio": None,
        "expectancy_r": None, "position_count": None, "avg_monthly_entries": None, "total_tax_krw": None,
        "limit_rejections": None, "yearly_pct": qqq_yearly,
    }

    print("[p5-4] 4장 실력인지 운인지(K1) ...", flush=True)
    k1_mc = monte_carlo_luck_test(data, cfg, k1, start, end, n_draws=500, seed_base=1000, ckpt_prefix="mc_k1")
    k1_boot = _run_checkpointed("boot_k1", lambda: bootstrap_and_yearly(data, k1, benchmark_main))
    k1_diag = {"mc": k1_mc, "boot": k1_boot, "wf_positive_years": None}

    k2_mc = k2_boot = k2_diag = None
    if k2 is not None:
        print("[p5-4] 4장 실력인지 운인지(K2) ...", flush=True)
        k2_mc = monte_carlo_luck_test(data_atr, cfg, k2, start, end, n_draws=500, seed_base=2000, ckpt_prefix="mc_k2")
        k2_boot = _run_checkpointed("boot_k2", lambda: bootstrap_and_yearly(data_atr, k2, benchmark_main))
        k2_diag = {"mc": k2_mc, "boot": k2_boot, "wf_positive_years": None}

    print("[p5-4] 5장 Walk-Forward ...", flush=True)
    wf = walk_forward(data, data_atr, cfg, qqq_yearly)
    wf_positive_years_overall = sum(1 for w in wf if w["excess_b_pp"] > 0)
    k1_diag["wf_positive_years"] = wf_positive_years_overall
    if k2_diag is not None:
        k2_diag["wf_positive_years"] = wf_positive_years_overall

    print("[p5-4] 6장 최종 후보 결정 ...", flush=True)
    final_choice, decision_detail = decide_final_candidate(k1_diag, k2_diag)

    config_path = None
    if final_choice is not None:
        config_path = write_final_candidate(cfg, final_choice, sec1, sec2, chosen_mult, start, end, warmup_start)

    run_id = f"{date.today().isoformat()}_{bt.make_run_id(cfg, start, end)[-8:]}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(
        out_dir, cfg, b05, sec1, sec2, k1, k2, qqq_row, k1_diag, k2_diag, wf, final_choice, decision_detail, config_path, start, end
    )
    print(f"[p5-4] 결과: {out_dir}", flush=True)
    return out_dir


def write_final_candidate(cfg, final_choice, sec1, sec2, chosen_mult, start, end, warmup_start) -> Path:
    try:
        commit_hash = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit_hash = None

    payload = {
        "candidate": final_choice,
        "generated_at": date.today().isoformat(),
        "generated_from_commit": commit_hash,
        "note": "이 파일이 만들어질 때의 HEAD 해시다. 이 실행 결과를 담은 커밋은 이 해시의 바로 다음 커밋이다.",
        "data": {"warmup_start": warmup_start.isoformat(), "train_start": start.isoformat(), "train_end": end.isoformat()},
        "base": "B0.5(슬롯 5)",
        "params": {
            "max_slots": 5,
            "c2_rsi_whipsaw": {"lookback_days": 15, "max_crosses": 2},
            "e1_atr_trail_mult": chosen_mult if final_choice == "K2" else None,
        },
    }
    CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_OUT.write_text(yaml.dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return CONFIG_OUT


def write_report(out_dir, cfg, b05, sec1, sec2, k1, k2, qqq_row, k1_diag, k2_diag, wf, final_choice, decision_detail, config_path, start, end):
    lines = [f"# P5-4 실험 결과 — {start} ~ {end}", "", "## 1장: C2 견고성", "", "| N | K | 세후(a) | 세후(b) | MDD | 포지션수 |", "| --- | --- | --- | --- | --- | --- |"]
    for N in (10, 15, 20):
        for K in (2, 3):
            s = sec1["grid"][(N, K)]
            lines.append(f"| {N} | {K} | {_fmt_pct(s['cagr_a_pct'])} | {_fmt_pct(s['cagr_b_pct'])} | {s['mdd_pct']}% | {s['position_count']} |")
    r = sec1["resim"]
    lines += [
        "",
        f"### C2(15·2) 상위 10개 반사실: (a) {_fmt_pct(r['before_cagr_a_pct'])} → {_fmt_pct(r['after_cagr_a_pct'])} (낙폭 {r['drop_pp']}%p, "
        f"참고 B0.5는 20.38%→14.63%로 낙폭 5.75%p)",
        f"- 제외 종목: {', '.join(r['excluded_tickers'])}",
        "",
        "### C2(15·2)가 걸러낸 A1 신호의 R 분포(B0.5 기준, 걸렸다면의 실제 결과)",
    ]
    bs = sec1["blocked_signal_r"]
    if bs:
        lines.append(f"- {bs['count']}건, 평균 R {bs['mean_r']}, 중앙값 {bs['median_r']}, 승률 {bs['win_rate_pct']}%, 범위 [{bs['min_r']}, {bs['max_r']}]")
    else:
        lines.append("- 해당 신호 없음")

    lines += ["", "## 2장: E1 배수 확장", "", "| 배수 | 세후(a) | 세후(b) | MDD | ATR_TRAIL 발동 | 평균 R |", "| --- | --- | --- | --- | --- | --- |"]
    for m in sec2["mults"]:
        s = sec2["grid"][m]
        at = sec2["atr_trail_stats"][m]
        lines.append(f"| x{m} | {_fmt_pct(s['cagr_a_pct'])} | {_fmt_pct(s['cagr_b_pct'])} | {s['mdd_pct']}% | {at['count']}건 | {at['avg_r']} |")
    lines.append(f"\n- 평탄 구간 선택 배수: {sec2['chosen_mult'] if sec2['chosen_mult'] is not None else '없음(불안정, 채택 안 함)'}")

    lines += ["", "## 3장: 조합 결과 표", "", "| 설계 | 세전 | 세후(a) | 세후(b) | QQQ대비(a) | QQQ대비(b) | MDD | 샤프 | 승률 | 손익비 | 기대값R | 포지션 | 월평균진입 | 양도세 | 한도초과 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]

    def row(s):
        a_rel = (s["cagr_a_pct"] or 0) - (qqq_row["cagr_a_pct"] or 0)
        b_rel = (s["cagr_b_pct"] or 0) - (qqq_row["cagr_b_pct"] or 0)
        tax_s = f"{s['total_tax_krw']:,}" if s.get("total_tax_krw") is not None else "-"
        return (
            f"| {s['label']} | {_fmt_pct(s['pretax_cagr_pct'])} | {_fmt_pct(s['cagr_a_pct'])} | {_fmt_pct(s['cagr_b_pct'])} | "
            f"{a_rel:+.2f}%p | {b_rel:+.2f}%p | {s['mdd_pct']}% | {s['sharpe']} | {s['win_rate_pct']}% | "
            f"{s['payoff_ratio']} | {s['expectancy_r']} | {s['position_count']} | {s['avg_monthly_entries']} | {tax_s}원 | {s['limit_rejections']} |"
        )

    lines.append(row(qqq_row))
    lines.append(row(b05))
    lines.append(row(k1))
    if k2 is not None:
        lines.append(row(k2))
    else:
        lines.append("| K2 | 생략(2장에서 E1 불안정) | | | | | | | | | | | | | |")

    lines += ["", "## 4장: 실력인지 운인지", ""]
    for name, diag in (("K1", k1_diag), ("K2", k2_diag)):
        if diag is None:
            lines.append(f"### {name}: 생략")
            continue
        mc, boot = diag["mc"], diag["boot"]
        lines += [
            f"### {name}",
            f"- 무작위 진입 비교(500회): 실제 세후(a) {_fmt_pct(mc['real_cagr_a_pct'])} — 무작위 분포의 {mc['percentile']}백분위 → **{mc['verdict']}** "
            f"(무작위 분포: 평균 {mc['draw_mean_pct']}%, 5~95백분위 [{mc['draw_p5_pct']}%, {mc['draw_p95_pct']}%])",
            f"- 부트스트랩(20일 블록, 2,000회) 연 초과수익 95% 신뢰구간: [{boot['bootstrap'].get('ci_low_pct')}%, {boot['bootstrap'].get('ci_high_pct')}%] "
            f"— 0 포함 여부: {'포함(유의하지 않음)' if boot['bootstrap'].get('includes_zero') else '미포함(유의)'}",
            f"- 연도별 초과 수익: " + " · ".join(f"{y}: {v:+.1f}%p" for y, v in boot["excess_by_year"].items()),
            f"- 초과 수익 난 연도: {boot['positive_years']}/{boot['total_years']}",
            "",
        ]

    lines += ["", "## 5장: Walk-Forward", "", "| Train | Test | 선택 (N,K) | 선택 E1배수 | Test 세후(a) | Test 세후(b) | QQQ(b) | 초과(b) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for w in wf:
        lines.append(
            f"| ~{w['train_end']} | {w['test_year']} | {w['chosen_nk']} | {w['chosen_mult'] if w['chosen_mult'] is not None else '없음'} | "
            f"{_fmt_pct(w['test_cagr_a_pct'])} | {_fmt_pct(w['test_cagr_b_pct'])} | {_fmt_pct(w['qqq_cagr_b_pct'])} | {w['excess_b_pp']:+.2f}%p |"
        )
    nk_set = {w["chosen_nk"] for w in wf}
    mult_set = {w["chosen_mult"] for w in wf}
    lines.append(f"\n- 선택된 (N,K)가 창마다 흔들리는지: {'흔들림 — ' + str(nk_set) if len(nk_set) > 1 else '고정 — ' + str(nk_set)}")
    lines.append(f"- 선택된 E1 배수가 창마다 흔들리는지: {'흔들림 — ' + str(mult_set) if len(mult_set) > 1 else '고정 — ' + str(mult_set)}")

    lines += ["", "## 6장: 최종 후보", ""]
    if final_choice is None:
        lines.append(f"- 최종 후보: **없음** (K1·K2 모두 판정 기준 미달) — 세부: {decision_detail}")
    else:
        lines.append(f"- 최종 후보: **{final_choice}** — 판정 근거: {decision_detail}")
        lines.append(f"- 고정 파일: {config_path}")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
