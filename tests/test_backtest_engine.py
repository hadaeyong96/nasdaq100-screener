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
    broker = bt.Broker(cash_usd=0.0)
    broker.buy_qqqm(1000.0, price_usd=100.0, cfg=full_cfg)  # 10주, 원가 1000달러
    assert broker.qqqm_shares == pytest.approx(10.0)
    broker.sell_qqqm(500.0, price_usd=150.0, cfg=full_cfg, fx_rate=1_300.0, year=2021)  # 약 3.33주 매도
    assert broker.qqqm_shares == pytest.approx(10.0 - 500.0 / 150.0)
    gain = broker.realized_gain_by_year[2021]
    assert gain > 0  # 150 > 100 평단가이므로 이익이 나야 한다


def test_broker_buy_sell_qqqm_has_no_fx_conversion(full_cfg):
    """P5-1.1 3번: QQQM 매매는 달러 안에서만 일어나 환전(스프레드)이 없어야 한다 —
    수수료만 깎이고, 환율이 높든 낮든(호출 시 fx_rate를 안 받으므로) 결과가 같아야 한다."""
    broker = bt.Broker(cash_usd=1_000.0)
    broker.buy_qqqm(500.0, price_usd=100.0, cfg=full_cfg)
    commission_pct = full_cfg["backtest"]["costs"]["commission_buy_pct"]
    expected_cash = 1_000.0 - 500.0 * (1 + commission_pct / 100)
    assert broker.cash_usd == pytest.approx(expected_cash)


def test_broker_pay_capital_gains_tax_uses_correct_year_and_zero_when_under_deduction(full_cfg):
    broker = bt.Broker(cash_usd=0.0)
    broker.buy_qqqm(10_000.0, price_usd=100.0, cfg=full_cfg)
    broker.realized_gain_by_year[2020] = 1_000_000  # 250만 원 공제 이하 -> 세금 0
    paid = broker.pay_capital_gains_tax(2020, price_usd=100.0, fx_rate=1_300.0, cfg=full_cfg, settlement_year=2021)
    assert paid == 0.0
    assert broker.tax_log[-1]["tax_krw"] == 0.0


def test_broker_pay_capital_gains_tax_crossing_year_charges_settlement_year(full_cfg):
    """세금을 마련하려고 QQQM을 파는 행위 자체의 실현손익은 납부 연도(settlement_year)에 잡혀야 한다."""
    broker = bt.Broker(cash_usd=0.0)
    broker.buy_qqqm(10_000.0, price_usd=100.0, cfg=full_cfg)
    broker.realized_gain_by_year[2020] = 10_000_000  # 2020년 실현손익 -> 세금 발생
    broker.pay_capital_gains_tax(2020, price_usd=120.0, fx_rate=1_300.0, cfg=full_cfg, settlement_year=2021)
    assert 2020 not in broker.realized_gain_by_year or broker.realized_gain_by_year[2020] == 10_000_000  # 2020년분은 그대로
    assert 2021 in broker.realized_gain_by_year  # 파는 행위 자체의 손익은 2021년으로 잡힌다
    assert broker.tax_log[-1]["year"] == 2020
    assert broker.tax_log[-1]["tax_krw"] > 0


def test_broker_apply_costs_false_skips_commission(full_cfg):
    broker = bt.Broker(cash_usd=1_000.0, apply_costs=False)
    broker.buy_qqqm(500.0, price_usd=100.0, cfg=full_cfg)
    assert broker.cash_usd == pytest.approx(500.0)  # 수수료 없음


def test_broker_apply_tax_false_skips_withholding_and_capital_gains(full_cfg):
    broker = bt.Broker(cash_usd=0.0, apply_tax=False)
    net = broker.receive_dividend(100.0, cfg=full_cfg)
    assert net == pytest.approx(100.0)  # 원천징수 없음
    broker.buy_qqqm(10_000.0, price_usd=100.0, cfg=full_cfg)
    broker.realized_gain_by_year[2020] = 100_000_000
    paid = broker.pay_capital_gains_tax(2020, price_usd=100.0, fx_rate=1_300.0, cfg=full_cfg, settlement_year=2021)
    assert paid == 0.0


# ── P5-1.1 검산 테스트 ────────────────────────────────────────────────────


def test_benchmark_liquidated_leq_unrealized(full_cfg):
    """P5-1.1 1번: 세금까지 낸 (a) 청산 값이 세금 안 낸 (b) 미청산 값보다 클 수 없다
    (이전 버전은 (a) 계산에서 보유분 평가액을 이중으로 더해 이 부등식이 깨졌었다)."""
    idx = pd.bdate_range("2015-01-02", periods=500)
    close = pd.Series([100 + i * 0.5 for i in range(500)], index=idx)  # 꾸준히 올라 실현손익이 커지게
    qqq_df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=idx)
    fx_by_date = {d.date().isoformat(): 1_300.0 for d in idx}
    data = bt.BacktestData(
        indicator_map={}, dividends={}, checkpoints=[], fx_by_date=fx_by_date,
        universe_mode="CURRENT_CONSTITUENTS", survivorship_bias="TRUE", failed_tickers={}, data_gap={},
        qqq_df=qqq_df, qqq_dividends=pd.Series(dtype=float),
        cash_etf_df=qqq_df, cash_etf_dividends=pd.Series(dtype=float),
    )
    result = bt.simulate_benchmark(data, full_cfg, idx[0].date(), idx[-1].date())
    assert result.cagr_liquidated <= result.cagr_unrealized + 1e-9


def test_aggregate_positions_groups_partial_exits_into_one_position():
    """P5-1.1 2번: 1차·2차·3차 부분 청산을 하나의 포지션으로 묶어야 한다."""
    trades = [
        {"date": "2020-01-02", "ticker": "AAA", "position_id": 1, "side": "진입", "initial_risk_krw": 100_000, "pnl_krw": None},
        {"date": "2020-01-10", "ticker": "AAA", "position_id": 1, "side": "청산", "pnl_krw": -30_000},
        {"date": "2020-01-20", "ticker": "AAA", "position_id": 1, "side": "청산", "pnl_krw": 80_000},
        {"date": "2020-02-01", "ticker": "AAA", "position_id": 1, "side": "청산", "pnl_krw": 50_000},
        {"date": "2020-03-01", "ticker": "BBB", "position_id": 2, "side": "진입", "initial_risk_krw": 50_000, "pnl_krw": None},
    ]
    positions = bt.aggregate_positions(trades, still_open_position_ids={2})
    assert len(positions) == 1  # BBB는 아직 열려 있어 제외된다
    pos = positions[0]
    assert pos["position_id"] == 1
    assert pos["pnl_krw"] == pytest.approx(-30_000 + 80_000 + 50_000)
    assert pos["r"] == pytest.approx(100_000 / 100_000)


def test_compute_position_stats_expectancy_matches_formula():
    """P5-1.1 2번: 기대값 = 승률 × 평균 이익(R) − 패율 × |평균 손실(R)| 이 실제 계산과 일치해야 한다."""
    positions = [
        {"r": 2.0, "pnl_krw": 200_000, "ticker": "A"},
        {"r": 1.5, "pnl_krw": 150_000, "ticker": "B"},
        {"r": -1.0, "pnl_krw": -100_000, "ticker": "C"},
        {"r": -0.5, "pnl_krw": -50_000, "ticker": "D"},
    ]
    stats = bt.compute_position_stats(positions)
    win_rate, loss_rate = 0.5, 0.5
    avg_win_r, avg_loss_r = (2.0 + 1.5) / 2, (-1.0 - 0.5) / 2
    expected = win_rate * avg_win_r - loss_rate * abs(avg_loss_r)
    assert stats["expectancy_r"] == pytest.approx(round(expected, 2))
    assert stats["win_rate_pct"] == 50.0
    assert stats["max_loss_r"] == -1.0


def test_splice_pre_inception_series_is_continuous_at_boundary():
    """P5-1.1 6번(회귀 방지): 접합 경계 앞뒤 가격이 튀지 않아야 한다 — 실제 QQQM 접합에서
    가격이 하루 만에 반토막 나는 회귀가 있었다."""
    idx_all = pd.bdate_range("2015-01-02", periods=1500)
    proxy_close = pd.Series([100.0 + i for i in range(1500)], index=idx_all)
    proxy_df = pd.DataFrame(
        {"open": proxy_close, "high": proxy_close + 1, "low": proxy_close - 1, "close": proxy_close}, index=idx_all
    )

    real_idx = idx_all[1000:]
    real_close = pd.Series([50.0 + i for i in range(len(real_idx))], index=real_idx)  # 완전히 다른 가격대
    real_df = pd.DataFrame(
        {"open": real_close, "high": real_close + 1, "low": real_close - 1, "close": real_close}, index=real_idx
    )

    spliced = bt.splice_pre_inception_series(proxy_df, real_df, proxy_expense_ratio_pct=0.20, real_expense_ratio_pct=0.15)
    assert spliced.index.is_monotonic_increasing
    assert not spliced.index.duplicated().any()

    boundary = real_idx[0]
    prev_date = idx_all[999]
    before = spliced.loc[prev_date, "close"]
    after = spliced.loc[boundary, "close"]
    assert after == pytest.approx(real_close.iloc[0])  # 접합일 자체는 실제 값 그대로
    assert abs(before / after - 1) < 0.05  # 하루 전후 괴리가 크면(예전처럼 반토막) 안 된다


def test_simulate_portfolio_produces_closeable_positions_with_initial_risk(full_cfg):
    """회귀 방지: simulate_portfolio가 만드는 진입 행에 initial_risk_krw가 실제로 채워져
    있어야 aggregate_positions가 포지션을 청산 완료로 인식한다 — 예전에는 이 값을
    open_positions에만 저장하고 trades 행에는 안 남겨 포지션이 전부 "열린 채로" 잡혔다."""
    n = 400
    rng = np.random.default_rng(11)
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
    result = bt.simulate_portfolio(data, full_cfg, idx[0].date(), idx[-1].date())
    entry_rows = [t for t in result.trades if t["side"] == "진입" and t.get("initial_risk_krw")]
    assert entry_rows, "새 포지션 진입 행에 initial_risk_krw가 채워져야 한다"

    positions = bt.aggregate_positions(result.trades, result.still_open_position_ids)
    assert positions, "완전히 청산된 포지션이 하나도 안 잡히면 회귀 버그가 되살아난 것이다"
    for p in positions:
        assert p["initial_risk_krw"]
        assert p["r"] is not None
