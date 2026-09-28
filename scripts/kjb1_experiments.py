"""KJB-1 실험 스크립트 (docs/kjb1_instructions.md 2번, configs/kjb1_preregistration.yaml).

일회성 리포트 스크립트 (CLAUDE.md scripts/). engine/backtest.py(B0.5)·engine/portfolio.py
(P0)·engine/kjb_satellite.py(KJB 위성)의 순수 시뮬레이션 함수를 그대로 재사용하고,
여기서는 조합(코어70+위성30)·무작위 진입 500회·보고만 오케스트레이션한다.

실행:
    python -u -m scripts.kjb1_experiments                 전체(본 실험 + 무작위 500회)
    python -u -m scripts.kjb1_experiments --random-only    체크포인트에서 무작위 진입만 이어서
    python -u -m scripts.kjb1_experiments --skip-random    무작위 진입 없이 본 실험만(빠른 확인용)

체크포인트: outputs/backtest/kjb1_checkpoint/random_entries.json에 완료된 시드의
결과를 누적 저장한다 — 중간에 죽어도 이어서 돌릴 수 있다. 진행 확인:
    python -c "import json; d=json.load(open('outputs/backtest/kjb1_checkpoint/random_entries.json', encoding='utf-8')); print(len(d), '/ 500')"
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import sys
from dataclasses import asdict
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

from data import sectors  # noqa: E402
from data import market_calendar  # noqa: E402
from engine import backtest as bt  # noqa: E402
from engine import kjb_satellite as ks  # noqa: E402
from engine import portfolio as pf  # noqa: E402

OUT_ROOT = bt.OUTPUT_ROOT / "kjb1"
CHECKPOINT_DIR = bt.OUTPUT_ROOT / "kjb1_checkpoint"
CHECKPOINT_PATH = CHECKPOINT_DIR / "random_entries.json"
PREREG_PATH = ROOT / "configs" / "kjb1_preregistration.yaml"
RANDOM_RUNS = 500


def load_preregistration() -> dict:
    with open(PREREG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_kjb_cfg(prereg: dict) -> dict:
    ms = prereg["main_settings"]
    return {
        "dollar_volume_multiplier_min": ms["signal"]["dollar_volume_multiplier_min"],
        "big_candle_pct_min": ms["signal"]["big_candle_pct_min"],
        "relative_return_window_days": ms["signal"]["relative_return_window_days"],
        "first_score_window_days": ms["signal"]["first_score_window_days"],
        "hold_trading_days": ms["exit"]["hold_trading_days"],
        "cooldown_trading_days": ms["reentry"]["cooldown_trading_days"],
        "max_concurrent_positions": ms["weighting"]["max_concurrent_positions"],
        "max_positions_per_sector": ms["weighting"]["max_positions_per_sector"],
        "per_position_fraction_of_satellite": ms["weighting"]["per_position_fraction_of_satellite"],
    }


def scaled_cfg(cfg: dict, fraction: float) -> dict:
    out = copy.deepcopy(cfg)
    out["backtest"]["total_krw"] = cfg["backtest"]["total_krw"] * fraction
    return out


def combine_equity_rows(rows_a: list[dict], rows_b: list[dict]) -> list[dict]:
    """두 결과의 equity_rows를 날짜로 맞춰 total_krw를 더한다(coreQQQM70 + 위성30 등)."""
    by_date_b = {r["date"]: r for r in rows_b}
    combined = []
    for ra in rows_a:
        rb = by_date_b.get(ra["date"])
        if rb is None:
            continue
        combined.append({
            "date": ra["date"],
            "qqqm_value_krw": ra.get("qqqm_value_krw", 0) + rb.get("qqqm_value_krw", 0),
            "positions_value_krw": ra.get("positions_value_krw", 0) + rb.get("positions_value_krw", 0),
            "cash_krw": ra.get("cash_krw", 0) + rb.get("cash_krw", 0),
            "total_krw": ra["total_krw"] + rb["total_krw"],
        })
    return combined


def run_main_scenarios(cfg: dict, prereg: dict, start: date, end: date, warmup_start: date):
    print("[kjb1] 시세·구성 종목 데이터 준비 중 (캐시 재사용)...", flush=True)
    data = bt.prepare_data(cfg, warmup_start, end)

    print(f"[kjb1] 섹터 정보(GICS) 받는 중 — {len(data.all_needed_tickers)}종목...", flush=True)
    sector_by_ticker, failed_sectors = sectors.get_sectors(sorted(data.all_needed_tickers))
    if failed_sectors:
        print(f"[kjb1] 섹터를 못 받은 종목 {len(failed_sectors)}개: {failed_sectors[:20]}", flush=True)

    kjb_cfg = build_kjb_cfg(prereg)

    print("[kjb1] 대기 자금(DTB3) 일별 수익률 준비 중...", flush=True)
    trading_days = [d.date() for d in market_calendar.trading_days_between(start, end)]
    reserve_daily_rate = ks.prepare_reserve_daily_rate(trading_days, start, end)

    cfg_30 = scaled_cfg(cfg, 0.30)
    cfg_70 = scaled_cfg(cfg, 0.70)

    print("[kjb1] KJB 위성(30%) 시뮬레이션 중...", flush=True)
    kjb_result, admission_log = ks.simulate_kjb_satellite(
        data, sector_by_ticker, cfg_30, kjb_cfg, start, end, reserve_daily_rate=reserve_daily_rate
    )

    print("[kjb1] B0.5(30%, 비교용) 시뮬레이션 중...", flush=True)
    b05_result = bt.simulate_portfolio(data, cfg_30, start, end, apply_costs=True, apply_tax=True, max_slots=5)

    print("[kjb1] 코어 QQQM 70%(P0) 시뮬레이션 중...", flush=True)
    portfolio_data_70 = pf.prepare_data(cfg_70, start, end)
    core70_result = pf.simulate_portfolio(portfolio_data_70, cfg_70, start, end, candidate="P0")

    print("[kjb1] QQQ·P0(100%, 기준선) 시뮬레이션 중...", flush=True)
    qqq_result = bt.simulate_benchmark(data, cfg, start, end)
    portfolio_data_100 = pf.prepare_data(cfg, start, end)
    p0_result = pf.simulate_portfolio(portfolio_data_100, cfg, start, end, candidate="P0")

    return {
        "data": data, "sector_by_ticker": sector_by_ticker, "failed_sectors": failed_sectors,
        "kjb_cfg": kjb_cfg, "reserve_daily_rate": reserve_daily_rate,
        "cfg_30": cfg_30, "cfg_70": cfg_70,
        "kjb_result": kjb_result, "admission_log": admission_log,
        "b05_result": b05_result, "core70_result": core70_result,
        "qqq_result": qqq_result, "p0_result": p0_result,
        "portfolio_data_70": portfolio_data_70, "portfolio_data_100": portfolio_data_100,
    }


def summarize_combo(core70_result, portfolio_data_70, satellite_result, data, cfg_70, cfg_30, cfg_full, start, end, label: str, satellite_is_kjb: bool) -> dict:
    combined_equity = combine_equity_rows(core70_result.equity_rows, satellite_result.equity_rows)
    equity_metrics = bt.compute_equity_metrics(combined_equity)

    core_a = pf.compute_posttax_a(core70_result, portfolio_data_70, cfg_70, end)
    # KJB 위성은 PortfolioBroker(대기 자금=reserve_usd)를 쓰고, B0.5는 engine.backtest.Broker
    # (qqqm_shares)를 쓴다 — 브로커 모양이 달라 청산 함수도 다르다(엔진마다 맞는 것을 쓴다).
    sat_a = (
        ks.compute_liquidated_value_krw(satellite_result, data, cfg_30, end)
        if satellite_is_kjb
        else bt.compute_liquidated_cagr(satellite_result, data, cfg_30, start, end)
    )

    total_krw = cfg_full["backtest"]["total_krw"]
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    combined_liquidated = (core_a.get("liquidated_value_krw") or 0) + (sat_a.get("liquidated_value_krw") or 0)
    cagr_a = (combined_liquidated / total_krw) ** (1 / years) - 1 if years > 0 and combined_liquidated > 0 else None

    positions = bt.aggregate_positions(satellite_result.trades, satellite_result.still_open_position_ids)
    position_stats = bt.compute_position_stats(positions)

    return {
        "label": label,
        "pretax_or_b_cagr_pct": equity_metrics.get("cagr_pct"),
        "cagr_a_pct": round(cagr_a * 100, 2) if cagr_a is not None else None,
        "mdd_pct": equity_metrics.get("mdd_pct"),
        "sharpe": equity_metrics.get("sharpe"),
        "position_count": position_stats.get("count", 0),
        "win_rate_pct": position_stats.get("win_rate_pct"),
        "payoff_ratio": position_stats.get("payoff_ratio"),
        "expectancy_r": position_stats.get("expectancy_r"),
        "combined_equity_rows": combined_equity,
    }


def summarize_satellite_alone(result, data, cfg_30, start, end, label: str) -> dict:
    """위성 단독(코어 70% QQQM과 섞지 않은) 성과 — 무작위 진입 비교와 같은 기준(위성
    30% 자금만)이라 percentile 비교는 반드시 이 값을 써야 한다(코어를 섞은 조합
    CAGR과 비교하면 코어의 강세장 수익이 섞여 들어가 신호 자체의 기여를 왜곡한다)."""
    equity_metrics = bt.compute_equity_metrics(result.equity_rows)
    liquidated = ks.compute_liquidated_value_krw(result, data, cfg_30, end)
    total_krw = cfg_30["backtest"]["total_krw"]
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    liquidated_krw = liquidated.get("liquidated_value_krw")
    cagr_a = round(((liquidated_krw / total_krw) ** (1 / years) - 1) * 100, 2) if liquidated_krw and years > 0 else None

    positions = bt.aggregate_positions(result.trades, result.still_open_position_ids)
    position_stats = bt.compute_position_stats(positions)
    return {
        "label": label,
        "pretax_or_b_cagr_pct": equity_metrics.get("cagr_pct"),
        "cagr_a_pct": cagr_a,
        "mdd_pct": equity_metrics.get("mdd_pct"),
        "sharpe": equity_metrics.get("sharpe"),
        "position_count": position_stats.get("count", 0),
        "win_rate_pct": position_stats.get("win_rate_pct"),
        "payoff_ratio": position_stats.get("payoff_ratio"),
        "expectancy_r": position_stats.get("expectancy_r"),
    }


def run_random_entry_worker(seed: int) -> dict:
    """Pool worker — 전역(_WORKER_*)은 initializer가 채운다."""
    result = ks.simulate_random_entry(
        _WORKER_DATA, _WORKER_SECTORS, _WORKER_CFG, _WORKER_KJB_CFG, _WORKER_START, _WORKER_END,
        _WORKER_ADMISSION_LOG, seed, signals=_WORKER_SIGNALS, reserve_daily_rate=_WORKER_RESERVE_RATE,
    )
    liquidated = ks.compute_liquidated_value_krw(result, _WORKER_DATA, _WORKER_CFG, _WORKER_END)
    liquidated_krw = liquidated.get("liquidated_value_krw")
    total_krw = _WORKER_CFG["backtest"]["total_krw"]
    years = (pd.Timestamp(_WORKER_END) - pd.Timestamp(_WORKER_START)).days / 365.25
    cagr_pct = round(((liquidated_krw / total_krw) ** (1 / years) - 1) * 100, 2) if liquidated_krw and years > 0 else None
    return {"seed": seed, "cagr_a_pct": cagr_pct}


def _init_worker(data, sector_by_ticker, cfg, kjb_cfg, start, end, admission_log, signals, reserve_daily_rate):
    global _WORKER_DATA, _WORKER_SECTORS, _WORKER_CFG, _WORKER_KJB_CFG, _WORKER_START, _WORKER_END
    global _WORKER_ADMISSION_LOG, _WORKER_SIGNALS, _WORKER_RESERVE_RATE
    _WORKER_DATA, _WORKER_SECTORS, _WORKER_CFG, _WORKER_KJB_CFG = data, sector_by_ticker, cfg, kjb_cfg
    _WORKER_START, _WORKER_END, _WORKER_ADMISSION_LOG = start, end, admission_log
    _WORKER_SIGNALS, _WORKER_RESERVE_RATE = signals, reserve_daily_rate


def run_random_entries(data, sector_by_ticker, cfg_30, kjb_cfg, start, end, admission_log, reserve_daily_rate, seed_base: int, n_runs: int = RANDOM_RUNS) -> list[dict]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    done: dict[int, dict] = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            for row in json.load(f):
                done[row["seed"]] = row

    signals = ks.build_price_only_signals(data)
    todo = [seed_base + i for i in range(n_runs) if (seed_base + i) not in done]
    print(f"[kjb1] 무작위 진입: 이미 완료 {len(done)} / 남은 것 {len(todo)}", flush=True)

    if todo:
        with multiprocessing.Pool(
            processes=min(multiprocessing.cpu_count(), 8),
            initializer=_init_worker,
            initargs=(data, sector_by_ticker, cfg_30, kjb_cfg, start, end, admission_log, signals, reserve_daily_rate),
        ) as pool:
            for i, row in enumerate(pool.imap_unordered(run_random_entry_worker, todo), start=1):
                done[row["seed"]] = row
                if i % 10 == 0 or i == len(todo):
                    print(f"[kjb1] 무작위 진입 진행: {len(done)} / {n_runs}", flush=True)
                    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                        json.dump(list(done.values()), f, ensure_ascii=False, indent=0)
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(list(done.values()), f, ensure_ascii=False, indent=0)

    return [done[s] for s in sorted(done) if s < seed_base + n_runs][:n_runs]


def percentile_rank(value: float, distribution: list[float]) -> float:
    if not distribution:
        return float("nan")
    below = sum(1 for v in distribution if v <= value)
    return round(below / len(distribution) * 100, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-only", action="store_true")
    parser.add_argument("--skip-random", action="store_true")
    args = parser.parse_args()

    cfg = bt.load_config()
    prereg = load_preregistration()
    start = date.fromisoformat(prereg["main_settings"]["period"]["start"])
    end = date.fromisoformat(prereg["main_settings"]["period"]["end"])
    warmup_start = bt._parse_date(cfg["backtest"]["warmup_start"])

    scenarios = run_main_scenarios(cfg, prereg, start, end, warmup_start)

    combo_kjb = summarize_combo(
        scenarios["core70_result"], scenarios["portfolio_data_70"], scenarios["kjb_result"], scenarios["data"],
        scenarios["cfg_70"], scenarios["cfg_30"], cfg, start, end, "코어70+KJB위성30", satellite_is_kjb=True,
    )
    combo_b05 = summarize_combo(
        scenarios["core70_result"], scenarios["portfolio_data_70"], scenarios["b05_result"], scenarios["data"],
        scenarios["cfg_70"], scenarios["cfg_30"], cfg, start, end, "코어70+B0.5위성30", satellite_is_kjb=False,
    )

    satellite_alone = summarize_satellite_alone(
        scenarios["kjb_result"], scenarios["data"], scenarios["cfg_30"], start, end, "KJB 위성 단독(30%)",
    )

    print("\n[kjb1] === 본 실험 결과(요약) ===", flush=True)
    for c in (combo_kjb, combo_b05, satellite_alone):
        print(f"  {c['label']}: (b)={c['pretax_or_b_cagr_pct']}% (a)={c['cagr_a_pct']}% MDD={c['mdd_pct']}% "
              f"Sharpe={c['sharpe']} 포지션={c['position_count']} 승률={c['win_rate_pct']}% "
              f"손익비={c['payoff_ratio']} 기대값R={c['expectancy_r']}", flush=True)

    random_results = []
    if not args.skip_random:
        seed = prereg["comparisons"]["random_entry"]["seed"]
        random_results = run_random_entries(
            scenarios["data"], scenarios["sector_by_ticker"], scenarios["cfg_30"], scenarios["kjb_cfg"],
            start, end, scenarios["admission_log"], scenarios["reserve_daily_rate"], seed,
        )
        dist = [r["cagr_a_pct"] for r in random_results if r["cagr_a_pct"] is not None]
        # 무작위 진입도 위성(30%)만 돌린 결과이므로, 반드시 위성 단독과 비교해야 한다
        # (코어를 섞은 조합 CAGR과 비교하면 코어의 강세장 수익이 섞여 들어가 왜곡된다).
        pct = percentile_rank(satellite_alone["cagr_a_pct"], dist) if satellite_alone["cagr_a_pct"] is not None else None
        print(f"\n[kjb1] 무작위 진입 {len(dist)}회 분포에서 KJB 위성 단독(30%, 세후 a) 백분위: {pct}", flush=True)

    out_dir = OUT_ROOT / bt.make_run_id(cfg, start, end)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scenarios["kjb_result"].trades).to_csv(out_dir / "trades_kjb.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(scenarios["admission_log"]).to_csv(out_dir / "admission_log.csv", index=False, encoding="utf-8-sig")
    if random_results:
        pd.DataFrame(random_results).to_csv(out_dir / "random_entries.csv", index=False, encoding="utf-8-sig")
    print(f"\n[kjb1] 결과 저장: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
