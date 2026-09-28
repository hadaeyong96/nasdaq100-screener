"""scripts/kjb1_experiments.py의 scaled_cfg 테스트 (KJB-1.1 회귀 방지). 네트워크 없이
합성 데이터로 돈다.

발견된 버그: 위성(예: 코어70+B0.5위성30)에 backtest.total_krw만 fraction만큼
줄인 cfg를 줬더니, core.sizing.slot_krw·strategy_limit_krw·funding_qty가 실제로는
account.total_krw(라이브 계좌 기본값, backtest.total_krw와 원래부터 다른 값)를
기준으로 슬롯·위험 상한을 계산해 — 위성이 자기 자본(scaled backtest.total_krw)의
최대 400%가 넘는 포지션을 잡고 현금이 -2500만원까지 마이너스로 내려갔다(실제
2016~2021 데이터로 확인). account.total_krw도 같이 fraction만큼만 줄이는 것으로는
부족했다(원래 계좌 간 2.5배 비율이 남아 여전히 121.8%까지 나옴) — account.total_krw를
backtest.total_krw와 정확히 같게 맞춰야 plan.strategy_limit_pct(60%)가 실제로
위성 자기 자본의 60%가 되어 노출이 100% 아래로 묶인다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.indicators import compute_indicators
from engine import backtest as bt
from scripts.kjb1_experiments import scaled_cfg


@pytest.fixture
def full_cfg(cfg):
    """account.total_krw(1억)과 backtest.total_krw(4천만)가 원래부터 다른, 실제
    config.yaml과 같은 비율의 cfg (test_backtest_engine.py의 full_cfg와 같은 모양)."""
    out = {**cfg}
    out["account"] = {"total_krw": 100_000_000}
    out["plan"] = {"strategy_limit_pct": 60, "cash_buffer_pct": 5, "max_slots": 5}
    out["backtest"] = {
        "total_krw": 40_000_000,
        "costs": {"commission_buy_pct": 0.07, "commission_sell_pct": 0.07, "fx_spread_pct": 0.1},
        "tax": {"capital_gains_deduction_krw": 2_500_000, "capital_gains_rate_pct": 22, "dividend_withholding_pct": 15, "payment_month": 5},
    }
    return out


def _synthetic_backtest_data(full_cfg, n: int = 400, seed: int = 11) -> tuple[bt.BacktestData, pd.Timestamp, pd.Timestamp]:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.05, 1.2, size=n)
    close = np.clip(100 + np.cumsum(steps), 5, None)
    high = close + rng.uniform(0.1, 1.5, size=n)
    low = np.minimum(close - rng.uniform(0.1, 1.5, size=n), close - 0.01)
    open_ = low + rng.uniform(0, 1, size=n) * (high - low)
    volume = rng.integers(1_000_000, 5_000_000, size=n)
    idx = pd.bdate_range("2015-01-02", periods=n, name="date")
    raw = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
    df = compute_indicators(raw, full_cfg)

    fx_by_date = {d.date().isoformat(): 1_300.0 for d in idx}
    data = bt.BacktestData(
        indicator_map={"AAA": df}, dividends={"AAA": pd.Series(dtype=float)}, checkpoints=[], fx_by_date=fx_by_date,
        universe_mode="CURRENT_CONSTITUENTS", survivorship_bias="TRUE", failed_tickers={}, data_gap={},
        qqq_df=df, qqq_dividends=pd.Series(dtype=float), cash_etf_df=df, cash_etf_dividends=pd.Series(dtype=float),
    )
    return data, idx[0].date(), idx[-1].date()


def test_scaled_cfg_keeps_account_and_backtest_total_krw_equal(full_cfg):
    scaled = scaled_cfg(full_cfg, 0.30)
    assert scaled["backtest"]["total_krw"] == pytest.approx(full_cfg["backtest"]["total_krw"] * 0.30)
    assert scaled["account"]["total_krw"] == scaled["backtest"]["total_krw"]


def test_scaled_cfg_satellite_exposure_never_exceeds_100_percent_of_its_own_capital(full_cfg):
    """가장 중요한 회귀 방지 테스트: 위성(30%)에 실제로 신호가 나서 포지션을 잡아도,
    보유 종목 평가액이 그 위성 자신의 자금(scaled backtest.total_krw)의 100%를
    절대 넘으면 안 된다. plan.strategy_limit_pct=60%가 진짜 상한이라면 최대여도
    60% 근처여야 한다(현금 버퍼·슬롯 배분 때문에 정확히 60%는 아닐 수 있음)."""
    data, start, end = _synthetic_backtest_data(full_cfg)
    cfg_30 = scaled_cfg(full_cfg, 0.30)

    result = bt.simulate_portfolio(data, cfg_30, start, end, max_slots=5)
    positions = bt.aggregate_positions(result.trades, result.still_open_position_ids)
    assert positions, "합성 데이터에서 청산 완료 포지션이 하나도 안 나오면 이 테스트가 뭘 지키는지 확인할 수 없다"
    assert result.equity_rows, "equity_rows가 비어 있으면 노출을 확인할 수 없다"

    total_krw = cfg_30["backtest"]["total_krw"]
    max_exposure_pct = max(row["positions_value_krw"] / total_krw * 100 for row in result.equity_rows)
    min_cash_krw = min(row["cash_krw"] for row in result.equity_rows)

    assert max_exposure_pct <= 100.0, f"위성 자기 자본의 100%를 넘는 노출: {max_exposure_pct:.1f}%"
    assert min_cash_krw >= 0, f"현금이 마이너스로 내려갔다: {min_cash_krw:.0f}원"
