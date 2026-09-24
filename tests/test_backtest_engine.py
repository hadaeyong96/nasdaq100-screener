"""engine/backtest.py 테스트 (P5-1 6번). 네트워크 없이 합성 데이터로 돈다.

- 미래 데이터 방지: t일까지 자른 데이터로 계산한 지표·신호가 전체 데이터의 t일
  값과 같아야 한다.
- 라이브·백테스트 일치: 같은 날짜·설정에서 engine.daily와 engine.backtest가
  core.sizing.size_buy_signals에 넘기는 수량 계산이 같아야 한다.
- Broker: QQQM 평균원가법 매수·매도 시 실현손익 계산, 세금 연도 넘김 납부.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import state as st
from core.indicators import compute_indicators
from engine import backtest as bt
from engine import daily as ed


def make_df(n: int = 400, seed: int = 0, start: str = "2020-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.05, 1.2, size=n)
    close = 100 + np.cumsum(steps)
    close = np.clip(close, 5, None)
    high = close + rng.uniform(0.1, 1.5, size=n)
    low = close - rng.uniform(0.1, 1.5, size=n)
    low = np.minimum(low, close - 0.01)
    open_ = low + rng.uniform(0, 1, size=n) * (high - low)
    volume = rng.integers(1_000_000, 5_000_000, size=n)
    index = pd.bdate_range(start, periods=n, name="date")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


@pytest.fixture
def full_cfg(cfg):
    out = {**cfg}
    out["account"] = {"total_krw": 100_000_000}
    out["plan"] = {"strategy_limit_pct": 60, "cash_buffer_pct": 5, "max_slots": 8}
    out["backtest"] = {
        "total_krw": 40_000_000,
        "costs": {"commission_buy_pct": 0.07, "commission_sell_pct": 0.07, "fx_spread_pct": 0.1},
        "tax": {"capital_gains_deduction_krw": 2_500_000, "capital_gains_rate_pct": 22, "dividend_withholding_pct": 15, "payment_month": 5},
    }
    return out


def test_no_lookahead_indicators_match_when_truncated(full_cfg):
    """t일까지 자른 데이터로 계산한 지표(RSI·MACD)가 전체 데이터의 t일 값과 같아야 한다."""
    raw = make_df(n=400, seed=3)
    full = compute_indicators(raw, full_cfg)
    t = full.index[250]
    truncated = compute_indicators(raw.loc[:t], full_cfg)
    for col in ("rsi", "macd", "signal", "gc", "dc"):
        a, b = full.loc[t, col], truncated.loc[t, col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == pytest.approx(b), f"{col} differs when truncated: {a} vs {b}"


def test_no_lookahead_signal_detection_matches_when_truncated(full_cfg):
    """전체 데이터 vs t일까지 자른 데이터로, t일 A1 신호 판정이 같아야 한다."""
    raw = make_df(n=400, seed=7)
    full = compute_indicators(raw, full_cfg)
    t = full.index[300]
    truncated = compute_indicators(raw.loc[:t], full_cfg)

    state = st.init_state("TEST")
    cand_full = st.preview_new_entry(full, t, state, full_cfg, None)
    cand_trunc = st.preview_new_entry(truncated, t, state, full_cfg, None)
    assert bool(cand_full) == bool(cand_trunc)
    if cand_full:
        assert cand_full["price"] == pytest.approx(cand_trunc["price"])
        assert cand_full["score"] == cand_trunc["score"]


def test_live_and_backtest_sizing_agree_for_same_day(cfg):
    """같은 날 두 종목에서 실제 A1 신호가 나면, engine.daily와 engine.backtest가
    core.sizing.size_buy_signals에 같은 입력을 만들어(따라서 같은 수량을) 낸다."""
    from tests.test_state import make_df as make_signal_df

    full_cfg = {
        **cfg,
        "account": {"total_krw": 40_000_000},
        "plan": {"strategy_limit_pct": 60, "cash_buffer_pct": 5, "max_slots": 8},
    }
    fx_rate = 1_350.0

    # 두 종목 다 전일 RSI<30 -> 당일 RSI>=30(A1). 점수는 vol_ratio로 갈라 순위를 만든다.
    rows_a = [{"close": 100, "rsi": 20}, {"close": 100, "rsi": 32, "swing_low": 94, "vol_ratio": 2.0}]
    rows_b = [{"close": 50, "rsi": 20}, {"close": 50, "rsi": 32, "swing_low": 47, "vol_ratio": 1.5}]
    df_a = make_signal_df(rows_a)
    df_b = make_signal_df(rows_b)
    date_ = df_a.index[1]
    indicator_map = {"AAA": df_a, "BBB": df_b}
    states = {"AAA": st.init_state("AAA"), "BBB": st.init_state("BBB")}

    events_by_ticker = {}
    for ticker in ("AAA", "BBB"):
        events, states[ticker] = st.process_day(indicator_map[ticker], date_, states[ticker], full_cfg, earnings_date=None, new_entry_allowed=True)
        events_by_ticker[ticker] = events
    assert all(states[t]["state"] == "주문대기" for t in ("AAA", "BBB"))

    from core import sizing

    # engine.daily 쪽 계산
    live_signals, _ = ed._todays_buy_signals(events_by_ticker, states, full_cfg)
    live_held = ed._held_signals_snapshot(states, indicator_map, date_)
    live_sized = sizing.size_buy_signals(live_signals, live_held, full_cfg, fx_rate)

    # engine.backtest 쪽 계산 (같은 states 딕셔너리를 건드리므로 복사해서 쓴다)
    import copy

    bt_states = copy.deepcopy(states)
    bt_rejected: list = []
    bt._size_and_stash(events_by_ticker, bt_states, indicator_map, date_, full_cfg, fx_rate, bt_rejected, date_.date())

    for ticker in ("AAA", "BBB"):
        key = f"{ticker}-A1"
        live_qty = live_sized["rows"][key]["qty"]
        bt_qty = bt_states[ticker]["pending"]["sized_qty"]
        assert live_qty == bt_qty, f"{ticker}: live={live_qty} backtest={bt_qty}"


def test_broker_qqqm_average_cost_realizes_gain_on_sale(full_cfg):
    broker = bt.Broker(cash_krw=0.0)
    broker.buy_qqqm(1000.0, price_usd=100.0, fx_rate=1_300.0, cfg=full_cfg)  # 10주, 원가 1000달러
    assert broker.qqqm_shares == pytest.approx(10.0)
    broker.sell_qqqm(500.0, price_usd=150.0, fx_rate=1_300.0, cfg=full_cfg, year=2021)  # 약 3.33주 매도
    assert broker.qqqm_shares == pytest.approx(10.0 - 500.0 / 150.0)
    gain = broker.realized_gain_by_year[2021]
    assert gain > 0  # 150 > 100 평단가이므로 이익이 나야 한다


def test_broker_pay_capital_gains_tax_uses_correct_year_and_zero_when_under_deduction(full_cfg):
    broker = bt.Broker(cash_krw=0.0)
    broker.buy_qqqm(10_000.0, price_usd=100.0, fx_rate=1_300.0, cfg=full_cfg)
    broker.realized_gain_by_year[2020] = 1_000_000  # 250만 원 공제 이하 -> 세금 0
    paid = broker.pay_capital_gains_tax(2020, price_usd=100.0, fx_rate=1_300.0, cfg=full_cfg, settlement_year=2021)
    assert paid == 0.0
    assert broker.tax_log[-1]["tax_krw"] == 0.0


def test_broker_pay_capital_gains_tax_crossing_year_charges_settlement_year(full_cfg):
    """세금을 마련하려고 QQQM을 파는 행위 자체의 실현손익은 납부 연도(settlement_year)에 잡혀야 한다."""
    broker = bt.Broker(cash_krw=0.0)
    broker.buy_qqqm(10_000.0, price_usd=100.0, fx_rate=1_300.0, cfg=full_cfg)
    broker.realized_gain_by_year[2020] = 10_000_000  # 2020년 실현손익 -> 세금 발생
    broker.pay_capital_gains_tax(2020, price_usd=120.0, fx_rate=1_300.0, cfg=full_cfg, settlement_year=2021)
    assert 2020 not in broker.realized_gain_by_year or broker.realized_gain_by_year[2020] == 10_000_000  # 2020년분은 그대로
    assert 2021 in broker.realized_gain_by_year  # 파는 행위 자체의 손익은 2021년으로 잡힌다
    assert broker.tax_log[-1]["year"] == 2020
    assert broker.tax_log[-1]["tax_krw"] > 0
