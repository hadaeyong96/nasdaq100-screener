"""engine/backtest.py 테스트 (P5-1 6번). 네트워크 없이 합성 데이터로 돈다.

- 미래 데이터 방지: t일까지 자른 데이터로 계산한 지표·신호가 전체 데이터의 t일
  값과 같아야 한다.
- 라이브·백테스트 일치: 같은 날짜·설정에서 engine.daily와 engine.backtest가
  core.sizing.size_buy_signals에 넘기는 수량 계산이 같아야 한다.
- Broker: QQQM 평균원가법 매수·매도 시 실현손익 계산, 세금 연도 넘김 납부.
"""

from __future__ import annotations

from datetime import date

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


# ── P5-2 테스트 ──────────────────────────────────────────────────────────


def test_compute_sma_regime_flags_above_and_below():
    idx = pd.bdate_range("2015-01-02", periods=250)
    # 앞 절반은 평평하게(이평선과 같음), 급등 직후 며칠은 확실히 이평선 위가 되게 만든다.
    close = pd.Series([100.0] * 125 + [200.0] * 125, index=idx)
    qqq_df = pd.DataFrame({"close": close}, index=idx)
    regime = bt.compute_sma_regime(qqq_df, window=50)
    assert regime[idx[100].date().isoformat()] is False  # 평평한 구간: 종가 == 이평선
    assert regime[idx[130].date().isoformat()] is True  # 급등 직후: 이평선이 아직 100대라 종가(200) > 이평선
    assert idx[10].date().isoformat() not in regime  # 워밍업 구간(이평선 계산 불가)은 빠진다


def test_compute_cloud_regime_requires_close_above_cloud_and_future_yang():
    idx = pd.bdate_range("2015-01-02", periods=5)
    qqq_df = pd.DataFrame(
        {
            "close": [110.0, 90.0, 110.0, 110.0, float("nan")],
            "cloud_top": [100.0, 100.0, 100.0, 100.0, 100.0],
            "future_yang": [True, True, False, True, True],
        },
        index=idx,
    )
    regime = bt.compute_cloud_regime(qqq_df)
    assert regime[idx[0].date().isoformat()] is True  # 종가>구름, 양운
    assert regime[idx[1].date().isoformat()] is False  # 종가<구름
    assert regime[idx[2].date().isoformat()] is False  # 음운
    assert idx[4].date().isoformat() not in regime  # close가 NaN이면 판정 불가로 뺀다


def test_simulate_portfolio_max_slots_override_reduces_new_positions():
    """P5-2 A1: max_slots를 좁히면(cfg의 plan.max_slots 대신) 동시 보유 한도 초과로
    막히는 신규 진입이 늘어(또는 최소 줄지 않아)야 한다."""
    n = 500
    tickers = {}
    for i, t in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        r = np.random.default_rng(100 + i)
        steps = r.normal(0.05, 1.2, size=n)
        close = np.clip(100 + np.cumsum(steps), 5, None)
        high = close + r.uniform(0.1, 1.5, size=n)
        low = np.minimum(close - r.uniform(0.1, 1.5, size=n), close - 0.01)
        open_ = low + r.uniform(0, 1, size=n) * (high - low)
        volume = r.integers(1_000_000, 5_000_000, size=n)
        idx = pd.bdate_range("2015-01-02", periods=n, name="date")
        tickers[t] = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
    idx = tickers["AAA"].index

    cfg = _plan_cfg_base()
    indicator_map = {t: compute_indicators(df, cfg) for t, df in tickers.items()}
    fx_by_date = {d.date().isoformat(): 1_300.0 for d in idx}
    data = bt.BacktestData(
        indicator_map=indicator_map, dividends={t: pd.Series(dtype=float) for t in tickers}, checkpoints=[],
        fx_by_date=fx_by_date, universe_mode="CURRENT_CONSTITUENTS", survivorship_bias="TRUE", failed_tickers={}, data_gap={},
        qqq_df=indicator_map["AAA"], qqq_dividends=pd.Series(dtype=float),
        cash_etf_df=indicator_map["AAA"], cash_etf_dividends=pd.Series(dtype=float),
    )
    result_wide = bt.simulate_portfolio(data, cfg, idx[0].date(), idx[-1].date(), max_slots=4)
    result_narrow = bt.simulate_portfolio(data, cfg, idx[0].date(), idx[-1].date(), max_slots=1)
    entries_wide = sum(1 for t in result_wide.trades if t["side"] == "진입" and t.get("initial_risk_krw"))
    entries_narrow = sum(1 for t in result_narrow.trades if t["side"] == "진입" and t.get("initial_risk_krw"))
    limit_rejections_narrow = sum(1 for r in result_narrow.rejected if "한도 초과" in r["reason"])
    assert entries_narrow <= entries_wide
    assert limit_rejections_narrow > 0


def _plan_cfg_base() -> dict:
    import yaml

    with open(bt.ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = dict(cfg)
    cfg["account"] = {"total_krw": 100_000_000}
    cfg["plan"] = {"strategy_limit_pct": 60, "cash_buffer_pct": 5, "max_slots": 8}
    cfg["backtest"] = dict(cfg["backtest"])
    cfg["backtest"]["total_krw"] = 40_000_000
    return cfg


def test_regime_ok_blocks_new_entries_but_not_exits(full_cfg):
    """P5-2 B1·B2: regime_ok가 False인 날은 새 진입이 전혀 admitted되지 않아야 한다."""
    from tests.test_state import make_df as make_signal_df

    rows = [{"close": 100, "rsi": 20}, {"close": 100, "rsi": 32, "swing_low": 94}]
    df = make_signal_df(rows)
    date_ = df.index[1]
    indicator_map = {"AAA": df}
    fx_by_date = {d.date().isoformat(): 1_300.0 for d in df.index}
    data = bt.BacktestData(
        indicator_map=indicator_map, dividends={"AAA": pd.Series(dtype=float)}, checkpoints=[], fx_by_date=fx_by_date,
        universe_mode="CURRENT_CONSTITUENTS", survivorship_bias="TRUE", failed_tickers={}, data_gap={},
        qqq_df=df, qqq_dividends=pd.Series(dtype=float), cash_etf_df=df, cash_etf_dividends=pd.Series(dtype=float),
    )
    regime_ok = {date_.date().isoformat(): False}
    result = bt.simulate_portfolio(data, full_cfg, df.index[0].date(), df.index[-1].date(), regime_ok=regime_ok)
    assert not any(t["side"] == "진입" for t in result.trades)


def test_compute_regime_split_stats_separates_returns_and_positions():
    """P5-2 0장: 국면별(above/below) 수익률·승률·기대값을 정확히 나눠야 한다."""
    equity_rows = [
        {"date": "2020-01-01", "total_krw": 100_000},
        {"date": "2020-01-02", "total_krw": 110_000},  # above, +10%
        {"date": "2020-01-03", "total_krw": 99_000},  # below, -10%
        {"date": "2020-01-04", "total_krw": 108_900},  # above, +10%
    ]
    regime_ok = {"2020-01-02": True, "2020-01-03": False, "2020-01-04": True}
    positions = [
        {"position_id": 1, "ticker": "AAA", "opened_date": "2020-01-01", "pnl_krw": 1000, "r": 1.0},
        {"position_id": 2, "ticker": "BBB", "opened_date": "2020-01-03", "pnl_krw": -500, "r": -1.0},
    ]
    out = bt.compute_regime_split_stats(equity_rows, positions, regime_ok)
    assert out["above"]["days"] == 2
    assert out["above"]["cum_return_pct"] == pytest.approx(21.0, abs=0.1)  # 1.1 * 1.1 - 1
    assert out["below"]["days"] == 1
    assert out["below"]["cum_return_pct"] == pytest.approx(-10.0, abs=0.1)
    assert out["above"]["position_count"] == 0  # opened 2020-01-01은 regime_ok에 없어 어디에도 안 들어감
    assert out["below"]["position_count"] == 1
    assert out["below"]["win_rate_pct"] == 0.0


def test_compute_topN_excluded_cagr_reduces_final_equity():
    """P5-2 0-4번: 상위 n개 포지션의 손익을 빼면 CAGR이 낮아져야 한다(근사)."""
    positions = [{"ticker": f"T{i}", "pnl_krw": (10 - i) * 1_000_000, "opened_date": "2015-01-01", "closed_date": "2015-06-01", "position_id": i, "initial_risk_krw": 100_000, "r": 1.0} for i in range(15)]
    total_krw = 40_000_000
    final_total_krw = 80_000_000
    from datetime import date as date_cls

    out = bt.compute_topN_excluded_cagr(positions, final_total_krw, total_krw, date_cls(2015, 1, 1), date_cls(2021, 12, 31), n=10)
    full_years = (date_cls(2021, 12, 31) - date_cls(2015, 1, 1)).days / 365.25
    full_cagr = (final_total_krw / total_krw) ** (1 / full_years) - 1
    assert out["cagr_pct"] < full_cagr * 100
    assert out["excluded_sum_pnl_krw"] > 0
    assert len(out["excluded_tickers"]) == 10


# ── P5-3 테스트 ──────────────────────────────────────────────────────────


def _synthetic_universe(tickers: list, n: int = 500, seed_base: int = 200) -> dict:
    out = {}
    for i, t in enumerate(tickers):
        r = np.random.default_rng(seed_base + i)
        steps = r.normal(0.05, 1.2, size=n)
        close = np.clip(100 + np.cumsum(steps), 5, None)
        high = close + r.uniform(0.1, 1.5, size=n)
        low = np.minimum(close - r.uniform(0.1, 1.5, size=n), close - 0.01)
        open_ = low + r.uniform(0, 1, size=n) * (high - low)
        volume = r.integers(1_000_000, 5_000_000, size=n)
        idx = pd.bdate_range("2015-01-02", periods=n, name="date")
        out[t] = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
    return out


def test_compute_relative_strength_top_half_ranks_by_trailing_return():
    idx = pd.bdate_range("2015-01-02", periods=200, name="date")
    flat = pd.Series(100.0, index=idx)
    rising = pd.Series(100.0 + np.arange(200) * 0.5, index=idx)
    indicator_map = {
        "FLAT": pd.DataFrame({"open": flat, "high": flat, "low": flat, "close": flat, "volume": 1_000_000}, index=idx),
        "RISE": pd.DataFrame({"open": rising, "high": rising, "low": rising, "close": rising, "volume": 1_000_000}, index=idx),
    }
    out = bt.compute_relative_strength_top_half(indicator_map, [d.date() for d in idx[150:151]], months=6)
    key = idx[150].date().isoformat()
    assert "RISE" in out[key]
    assert "FLAT" not in out[key]


def test_compute_rsi_whipsaw_block_flags_after_second_cross():
    idx = pd.bdate_range("2015-01-02", periods=10, name="date")
    rsi = [20, 35, 20, 35, 20, 20, 20, 20, 20, 20]
    df = pd.DataFrame({"rsi": rsi}, index=idx)
    out = bt.compute_rsi_whipsaw_block({"AAA": df}, lookback=15, max_crosses=2)
    assert out[("AAA", idx[3].date().isoformat())] is True
    assert out[("AAA", idx[1].date().isoformat())] is False


def test_compute_volume_entry_filter_volume_mode():
    idx = pd.bdate_range("2015-01-02", periods=3, name="date")
    df = pd.DataFrame({"vol_ratio": [1.0, 2.0, 1.6]}, index=idx)
    out = bt.compute_volume_entry_filter({"AAA": df}, mode="volume")
    assert out[("AAA", idx[0].date().isoformat())] is False
    assert out[("AAA", idx[1].date().isoformat())] is True
    assert out[("AAA", idx[2].date().isoformat())] is True


def test_compute_volume_entry_filter_obv_mode_rises_with_up_trend():
    idx = pd.bdate_range("2015-01-02", periods=25, name="date")
    close = pd.Series(100.0 + np.arange(25) * 0.3, index=idx)
    df = pd.DataFrame({"close": close, "volume": 1_000_000}, index=idx)
    out = bt.compute_volume_entry_filter({"AAA": df}, mode="obv")
    assert out[("AAA", idx[24].date().isoformat())] is True


def test_entry_type_allowed_empty_blocks_all_new_entries(full_cfg):
    from tests.test_state import make_df as make_signal_df

    rows = [{"close": 100, "rsi": 20}, {"close": 100, "rsi": 32, "swing_low": 94}]
    df = make_signal_df(rows)
    indicator_map = {"AAA": df}
    fx_by_date = {d.date().isoformat(): 1_300.0 for d in df.index}
    data = bt.BacktestData(
        indicator_map=indicator_map, dividends={"AAA": pd.Series(dtype=float)}, checkpoints=[], fx_by_date=fx_by_date,
        universe_mode="CURRENT_CONSTITUENTS", survivorship_bias="TRUE", failed_tickers={}, data_gap={},
        qqq_df=df, qqq_dividends=pd.Series(dtype=float), cash_etf_df=df, cash_etf_dividends=pd.Series(dtype=float),
    )
    result = bt.simulate_portfolio(data, full_cfg, df.index[0].date(), df.index[-1].date(), entry_type_allowed=set())
    assert not any(t["side"] == "진입" for t in result.trades)


def test_blocked_new_entries_prevents_that_specific_fill():
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    raw = _synthetic_universe(tickers)
    cfg = _plan_cfg_base()
    indicator_map = {t: compute_indicators(df, cfg) for t, df in raw.items()}
    idx = raw["AAA"].index
    fx_by_date = {d.date().isoformat(): 1_300.0 for d in idx}
    data = bt.BacktestData(
        indicator_map=indicator_map, dividends={t: pd.Series(dtype=float) for t in tickers}, checkpoints=[],
        fx_by_date=fx_by_date, universe_mode="CURRENT_CONSTITUENTS", survivorship_bias="TRUE", failed_tickers={}, data_gap={},
        qqq_df=indicator_map["AAA"], qqq_dividends=pd.Series(dtype=float),
        cash_etf_df=indicator_map["AAA"], cash_etf_dividends=pd.Series(dtype=float),
    )
    baseline = bt.simulate_portfolio(data, cfg, idx[0].date(), idx[-1].date())
    first_entry = next(t for t in baseline.trades if t["side"] == "진입" and t.get("initial_risk_krw"))
    blocked = {(first_entry["ticker"], first_entry["date"])}
    result = bt.simulate_portfolio(data, cfg, idx[0].date(), idx[-1].date(), blocked_new_entries=blocked)
    still_present = any(
        t["side"] == "진입" and t["ticker"] == first_entry["ticker"] and t["date"] == first_entry["date"] and t.get("initial_risk_krw")
        for t in result.trades
    )
    assert not still_present


def test_atr_trail_mult_runs_and_does_not_widen_worst_loss():
    """P5-3 E1: 스모크 테스트 — 에러 없이 돌고, 청산 완료 포지션의 최대 손실(R)이
    기본(swing_low 손절만)보다 더 나빠지지는 않아야 한다(추적 손절은 손절가를
    더 타이트하게만 만든다)."""
    from core.indicators import compute_atr

    tickers = ["AAA", "BBB", "CCC"]
    raw = _synthetic_universe(tickers, seed_base=300)
    cfg = _plan_cfg_base()
    indicator_map = {}
    for t, df in raw.items():
        ind = compute_indicators(df, cfg)
        ind["atr"] = compute_atr(df, period=14)
        indicator_map[t] = ind
    idx = raw["AAA"].index
    fx_by_date = {d.date().isoformat(): 1_300.0 for d in idx}
    data = bt.BacktestData(
        indicator_map=indicator_map, dividends={t: pd.Series(dtype=float) for t in tickers}, checkpoints=[],
        fx_by_date=fx_by_date, universe_mode="CURRENT_CONSTITUENTS", survivorship_bias="TRUE", failed_tickers={}, data_gap={},
        qqq_df=indicator_map["AAA"], qqq_dividends=pd.Series(dtype=float),
        cash_etf_df=indicator_map["AAA"], cash_etf_dividends=pd.Series(dtype=float),
    )
    baseline = bt.simulate_portfolio(data, cfg, idx[0].date(), idx[-1].date())
    trailed = bt.simulate_portfolio(data, cfg, idx[0].date(), idx[-1].date(), atr_trail_mult=2.5)

    base_positions = bt.aggregate_positions(baseline.trades, baseline.still_open_position_ids)
    trail_positions = bt.aggregate_positions(trailed.trades, trailed.still_open_position_ids)
    base_worst = min((p["r"] for p in base_positions), default=0.0)
    trail_worst = min((p["r"] for p in trail_positions), default=0.0)
    assert trail_worst >= base_worst - 1e-6


def test_partial_tp_r_mult_sells_half_once_per_position():
    tickers = ["AAA", "BBB", "CCC"]
    raw = _synthetic_universe(tickers, seed_base=400)
    cfg = _plan_cfg_base()
    indicator_map = {t: compute_indicators(df, cfg) for t, df in raw.items()}
    idx = raw["AAA"].index
    fx_by_date = {d.date().isoformat(): 1_300.0 for d in idx}
    data = bt.BacktestData(
        indicator_map=indicator_map, dividends={t: pd.Series(dtype=float) for t in tickers}, checkpoints=[],
        fx_by_date=fx_by_date, universe_mode="CURRENT_CONSTITUENTS", survivorship_bias="TRUE", failed_tickers={}, data_gap={},
        qqq_df=indicator_map["AAA"], qqq_dividends=pd.Series(dtype=float),
        cash_etf_df=indicator_map["AAA"], cash_etf_dividends=pd.Series(dtype=float),
    )
    result = bt.simulate_portfolio(data, cfg, idx[0].date(), idx[-1].date(), partial_tp_r_mult=2.0)
    partials = [t for t in result.trades if t.get("stage") == "PARTIAL_TP"]
    seen_ids = set()
    for t in partials:
        assert t["side"] == "청산"
        assert t["qty"] > 0
        assert t["position_id"] not in seen_ids
        seen_ids.add(t["position_id"])


def test_compute_stock_sleeve_twr_usd_ignores_days_with_no_exposure_and_cashflow():
    """진입일의 순매수만큼은 수익률에서 빠지고, 보유 가치가 0인 날은 건너뛰어야 한다."""
    equity_rows = [
        {"date": "2020-01-01", "positions_value_krw": 0, "total_krw": 100_000},
        {"date": "2020-01-02", "positions_value_krw": 10_000, "total_krw": 100_000},  # 진입(순매수 10,000원어치) — 수익률 계산 제외 대상
        {"date": "2020-01-03", "positions_value_krw": 11_000, "total_krw": 100_000},  # +10% (10,000 -> 11,000, 매매 없음)
    ]
    trades = [{"date": "2020-01-02", "side": "진입", "price": 100.0, "qty": 100}]  # 10,000 KRW/1300 = ~7.69 USD 무시(간단화: fx=1)
    fx_by_date = {"2020-01-01": 1.0, "2020-01-02": 1.0, "2020-01-03": 1.0}
    out = bt.compute_stock_sleeve_twr_usd(equity_rows, trades, fx_by_date, annualize_basis=252)
    assert out["invested_days"] == 1  # 01-02(순매수 있어 prev_v=0으로 스킵)는 제외, 01-03만 포함
    assert "2020-01-03" in out["daily_returns"]
    assert out["daily_returns"]["2020-01-03"] == pytest.approx(0.1, abs=1e-6)


def test_compute_qqq_twr_over_days_usd_filters_to_given_days():
    idx = pd.bdate_range("2020-01-01", periods=5, name="date")
    close = pd.Series([100.0, 110.0, 121.0, 100.0, 90.0], index=idx)
    qqq_df = pd.DataFrame({"close": close}, index=idx)
    invested_days = {idx[2].date().isoformat()}  # 01-02 -> 01-03 구간(+10%)만 포함
    out = bt.compute_qqq_twr_over_days_usd(qqq_df, pd.Series(dtype=float), invested_days, annualize_basis=252)
    assert out["days"] == 1
    assert out["twr_annualized_pct"] is not None


def test_compute_position_weight_stats_basic():
    equity_rows = [
        {"date": "2020-01-01", "positions_value_krw": 0, "total_krw": 100},
        {"date": "2020-01-02", "positions_value_krw": 50, "total_krw": 100},
        {"date": "2020-01-03", "positions_value_krw": 20, "total_krw": 100},
    ]
    out = bt.compute_position_weight_stats(equity_rows)
    assert out["max_weight_pct"] == 50.0
    assert out["min_weight_pct"] == 0.0
    assert out["avg_weight_pct"] == pytest.approx((0 + 50 + 20) / 3, abs=0.01)


def test_compute_entry_type_breakdown_splits_a_and_b():
    trades = [
        {"position_id": 1, "side": "진입", "stage": "A1", "initial_risk_krw": 1000},
        {"position_id": 2, "side": "진입", "stage": "B", "initial_risk_krw": 2000},
    ]
    positions = [
        {"position_id": 1, "ticker": "AAA", "pnl_krw": 500, "r": 0.5},
        {"position_id": 2, "ticker": "BBB", "pnl_krw": -400, "r": -0.2},
    ]
    out = bt.compute_entry_type_breakdown(positions, trades)
    assert out["A형"]["count"] == 1
    assert out["A형"]["total_pnl_krw"] == 500
    assert out["B형"]["count"] == 1
    assert out["B형"]["total_pnl_krw"] == -400


def test_compute_exit_type_breakdown_averages_r_per_kind():
    trades = [
        {"position_id": 1, "side": "진입", "stage": "A1", "initial_risk_krw": 1000},
        {"position_id": 1, "side": "청산", "stage": "STOP", "pnl_krw": -1000},
        {"position_id": 2, "side": "진입", "stage": "A1", "initial_risk_krw": 2000},
        {"position_id": 2, "side": "청산", "stage": "STOP", "pnl_krw": -3000},
    ]
    out = bt.compute_exit_type_breakdown(trades)
    assert out["STOP"]["count"] == 2
    assert out["STOP"]["avg_r"] == pytest.approx((-1.0 + -1.5) / 2, abs=1e-6)


# ── P5-4 테스트 ──────────────────────────────────────────────────────────


def test_atr_trail_exit_labeled_atr_trail_not_stop():
    """P5-4 0-1번: 추적 손절이 실제로 발동한 청산은 STOP이 아니라 ATR_TRAIL로 기록돼야 한다."""
    from core.indicators import compute_atr

    tickers = ["AAA", "BBB", "CCC"]
    raw = _synthetic_universe(tickers, seed_base=500)
    cfg = _plan_cfg_base()
    indicator_map = {}
    for t, df in raw.items():
        ind = compute_indicators(df, cfg)
        ind["atr"] = compute_atr(df, period=14)
        indicator_map[t] = ind
    idx = raw["AAA"].index
    fx_by_date = {d.date().isoformat(): 1_300.0 for d in idx}
    data = bt.BacktestData(
        indicator_map=indicator_map, dividends={t: pd.Series(dtype=float) for t in tickers}, checkpoints=[],
        fx_by_date=fx_by_date, universe_mode="CURRENT_CONSTITUENTS", survivorship_bias="TRUE", failed_tickers={}, data_gap={},
        qqq_df=indicator_map["AAA"], qqq_dividends=pd.Series(dtype=float),
        cash_etf_df=indicator_map["AAA"], cash_etf_dividends=pd.Series(dtype=float),
    )
    trailed = bt.simulate_portfolio(data, cfg, idx[0].date(), idx[-1].date(), atr_trail_mult=0.3)
    atr_trail_trades = [t for t in trailed.trades if t.get("stage") == "ATR_TRAIL"]
    assert atr_trail_trades
    for t in atr_trail_trades:
        assert t["side"] == "청산"
        assert t["reason"] == "ATR 추적 손절"


def test_partial_tp_fills_use_next_day_open_price():
    """P5-4 0-2번: 분할 익절 체결가는 신호 다음날 시가여야 한다(다른 청산과 통일)."""
    tickers = ["AAA", "BBB", "CCC"]
    raw = _synthetic_universe(tickers, seed_base=400)
    cfg = _plan_cfg_base()
    indicator_map = {t: compute_indicators(df, cfg) for t, df in raw.items()}
    idx = raw["AAA"].index
    fx_by_date = {d.date().isoformat(): 1_300.0 for d in idx}
    data = bt.BacktestData(
        indicator_map=indicator_map, dividends={t: pd.Series(dtype=float) for t in tickers}, checkpoints=[],
        fx_by_date=fx_by_date, universe_mode="CURRENT_CONSTITUENTS", survivorship_bias="TRUE", failed_tickers={}, data_gap={},
        qqq_df=indicator_map["AAA"], qqq_dividends=pd.Series(dtype=float),
        cash_etf_df=indicator_map["AAA"], cash_etf_dividends=pd.Series(dtype=float),
    )
    result = bt.simulate_portfolio(data, cfg, idx[0].date(), idx[-1].date(), partial_tp_r_mult=2.0)
    partials = [t for t in result.trades if t.get("stage") == "PARTIAL_TP"]
    assert partials
    for t in partials:
        df = indicator_map[t["ticker"]]
        ts = pd.Timestamp(t["date"])
        assert ts in df.index
        assert t["price"] == pytest.approx(float(df.loc[ts, "open"]), rel=1e-6)


def test_select_flat_region_smallest_picks_smallest_stable_value():
    mults = [2.5, 3.0, 3.5, 4.0]
    metric = {2.5: 10.0, 3.0: 10.5, 3.5: 10.4, 4.0: 15.0}
    # 3.0·3.5는 서로(그리고 있는 이웃과) 1 이내 -> 평탄 구간. 2.5는 3.0과 0.5차이라 이웃 기준 평탄,
    # 4.0은 3.5와 4.6차이라 평탄 아님. 평탄 구간 중 가장 작은 값 = 2.5(3.0과 0.5차이로 평탄 조건 만족).
    out = bt.select_flat_region_smallest(mults, metric, tol_pp=1.0)
    assert out == 2.5


def test_select_flat_region_smallest_returns_none_when_no_flat_region():
    mults = [2.5, 3.0, 3.5]
    metric = {2.5: 5.0, 3.0: 20.0, 3.5: 40.0}
    out = bt.select_flat_region_smallest(mults, metric, tol_pp=1.0)
    assert out is None


def test_compute_block_bootstrap_ci_reproducible_and_detects_zero_excess():
    daily = [0.0] * 500  # 초과 수익이 항상 0이면 신뢰구간이 0을 포함해야 한다
    out1 = bt.compute_block_bootstrap_ci(daily, block_size=20, n_resamples=200, seed=42)
    out2 = bt.compute_block_bootstrap_ci(daily, block_size=20, n_resamples=200, seed=42)
    assert out1 == out2  # 같은 시드 -> 같은 결과
    assert out1["includes_zero"] is True


def test_compute_block_bootstrap_ci_excludes_zero_for_clear_positive_excess():
    daily = [0.01] * 500  # 매일 +1% 초과 수익 -> 연율화하면 확실히 양수, 신뢰구간이 0 위에 있어야 함
    out = bt.compute_block_bootstrap_ci(daily, block_size=20, n_resamples=200, seed=1)
    assert out["ci_low_pct"] > 0
    assert out["includes_zero"] is False


def test_find_first_technical_exit_detects_stop_before_later_e1():
    idx = pd.bdate_range("2021-01-04", periods=6, name="date")
    df = pd.DataFrame(
        {
            "close": [100.0, 90.0, 80.0, 70.0, 60.0, 50.0],
            "dc": [False, False, False, True, False, False],
            "rsi": [60.0, 55.0, 52.0, 48.0, 45.0, 40.0],
            "cloud_bot": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
            "chikou_broken": [False, False, False, False, False, False],
        },
        index=idx,
    )
    out = bt.find_first_technical_exit(df, idx[0], stop_price=85.0)
    assert out["kind"] == "STOP"
    assert out["date"] == idx[2]  # close 80 <= 85 이 첫 번째로 걸리는 날(idx[2])


def test_find_first_technical_exit_returns_none_when_nothing_triggers():
    idx = pd.bdate_range("2021-01-04", periods=4, name="date")
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0],
            "dc": [False, False, False, False],
            "rsi": [60.0, 61.0, 62.0, 63.0],
            "cloud_bot": [50.0, 50.0, 50.0, 50.0],
            "chikou_broken": [False, False, False, False],
        },
        index=idx,
    )
    out = bt.find_first_technical_exit(df, idx[0], stop_price=10.0)
    assert out is None


# ── P5-5 테스트: compute_yearly_universe_coverage ───────────────────────────


def _mk_price_df(dates: list) -> pd.DataFrame:
    idx = pd.DatetimeIndex(dates, name="date")
    return pd.DataFrame({"close": [100.0] * len(idx)}, index=idx)


def test_compute_yearly_universe_coverage_full_coverage_gives_zero_missing_ratio():
    from data import universe_history as uh

    days_2007 = pd.bdate_range("2007-01-02", "2007-12-31")
    changes: list = []  # 구성 변경 없음 -> AAA·BBB가 연중 내내 구성 종목
    checkpoints = uh.membership_checkpoints({"AAA", "BBB"}, changes, date(2007, 1, 1))
    indicator_map = {"AAA": _mk_price_df(days_2007), "BBB": _mk_price_df(days_2007)}

    out = bt.compute_yearly_universe_coverage(checkpoints, indicator_map, list(days_2007))
    assert out[2007]["constituent_count"] == 2
    assert out[2007]["priced_count"] == 2
    assert out[2007]["missing_ratio_pct"] == 0.0
    assert out[2007]["missing_members"] == []


def test_compute_yearly_universe_coverage_flags_ticker_with_no_price_data():
    from data import universe_history as uh

    days_2007 = pd.bdate_range("2007-01-02", "2007-12-31")
    changes: list = []
    checkpoints = uh.membership_checkpoints({"AAA", "BBB"}, changes, date(2007, 1, 1))
    indicator_map = {"AAA": _mk_price_df(days_2007)}  # BBB는 시세를 못 받음

    out = bt.compute_yearly_universe_coverage(checkpoints, indicator_map, list(days_2007))
    assert out[2007]["constituent_count"] == 2
    assert out[2007]["priced_count"] == 1
    assert out[2007]["missing_members"] == ["BBB"]
    assert out[2007]["missing_ratio_pct"] == 50.0  # 종목·일수 절반이 BBB(누락)


def test_compute_yearly_universe_coverage_partial_year_membership_change():
    """연중 구성이 바뀌면(BBB가 7월부터 CCC로 교체) 그 연도의 constituent_count에 둘 다 잡혀야 한다."""
    from data.universe_history import IndexChange, membership_checkpoints

    days_2007 = pd.bdate_range("2007-01-02", "2007-12-31")
    changes = [IndexChange(date=date(2007, 7, 1), added="CCC", removed="BBB")]
    checkpoints = membership_checkpoints({"AAA", "CCC"}, changes, date(2007, 1, 1))
    indicator_map = {
        "AAA": _mk_price_df(days_2007),
        "BBB": _mk_price_df(pd.bdate_range("2007-01-02", "2007-06-29")),
        "CCC": _mk_price_df(pd.bdate_range("2007-07-01", "2007-12-31")),
    }

    out = bt.compute_yearly_universe_coverage(checkpoints, indicator_map, list(days_2007))
    assert out[2007]["constituent_count"] == 3  # AAA·BBB·CCC 모두 그해 한 번이라도 구성 종목이었음
    assert out[2007]["missing_ratio_pct"] == 0.0  # 각자 재직 기간에는 시세가 다 있음
    assert out[2007]["members"] == ["AAA", "BBB", "CCC"]


def test_compute_yearly_universe_coverage_splits_by_calendar_year():
    from data import universe_history as uh

    days = list(pd.bdate_range("2007-01-02", "2008-12-31"))
    changes: list = []
    checkpoints = uh.membership_checkpoints({"AAA"}, changes, date(2007, 1, 1))
    indicator_map = {"AAA": _mk_price_df(pd.bdate_range("2007-01-02", "2007-12-31"))}  # 2008년은 시세 없음

    out = bt.compute_yearly_universe_coverage(checkpoints, indicator_map, days)
    assert out[2007]["missing_ratio_pct"] == 0.0
    assert out[2008]["missing_ratio_pct"] == 100.0
