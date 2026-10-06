"""L1 국면 신호·봉인·합성 QLD·목표 비중 시뮬레이터 테스트 (네트워크 없음, 합성 데이터)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import regime as rg
from core import synthetic_assets as sa
from engine import portfolio as pf


def _trading_days(start="2015-01-02", periods=600):
    return pd.bdate_range(start, periods=periods)


def _close(td, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.015, len(td)))), index=td)


def _unrate(months_start="2013-01-01", n=48, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(months_start, periods=n, freq="MS")
    return pd.Series(5 + np.cumsum(rng.normal(0, 0.2, n)), index=idx)


# ── 미래 데이터 방지 ───────────────────────────────────────────────────────


@pytest.mark.parametrize("cut", [250, 380, 520])
@pytest.mark.parametrize("candidate", ["L2", "L3", "L4"])
def test_signal_truncated_equals_full(candidate, cut):
    td = _trading_days()
    close = _close(td)
    unrate = _unrate()
    t = td[cut]

    def weights(close_s, days, months):
        above = rg.above_sma(close_s, 200)
        up = rg.monthly_above_avg(months, days, 12)
        return rg.target_weights(candidate, above, up)

    full = weights(close, td, unrate)
    cut_months = unrate[unrate.index <= t]  # t일까지 존재한 월만(발표는 더 늦다)
    trunc = weights(close.loc[:t], td[td <= t], cut_months)
    pd.testing.assert_frame_equal(full.loc[:t], trunc)


def test_above_sma_equal_counts_as_below_and_warmup_nan():
    td = _trading_days(periods=5)
    close = pd.Series([1.0, 1.0, 1.0, 2.0, 1.0], index=td)
    out = rg.above_sma(close, 3)
    assert pd.isna(out.iloc[0]) and pd.isna(out.iloc[1])
    assert out.iloc[2] is False or out.iloc[2] == False  # noqa: E712 — 같으면 위가 아님
    assert out.iloc[3] == True  # noqa: E712
    assert out.iloc[4] == False  # noqa: E712


# ── 실업률 발표 지연 ─────────────────────────────────────────────────────


def test_unrate_month_not_used_before_m_plus_2_first_trading_day():
    td = pd.bdate_range("2020-01-01", "2020-12-31")
    months = pd.date_range("2019-01-01", "2020-12-01", freq="MS")
    vals = pd.Series(3.0, index=months)
    vals[pd.Timestamp("2020-03-01")] = 9.0  # 3월 급등 — 5월 첫 거래일(2020-05-01)부터만 보여야 한다
    up = rg.monthly_above_avg(vals, td, 12)
    assert not up.loc[:"2020-04-30"].any()
    assert up.loc["2020-05-01"]
    rel = rg.monthly_release_dates(pd.DatetimeIndex([pd.Timestamp("2020-03-01")]), td)
    assert rel.iloc[0] == pd.Timestamp("2020-05-01")


def test_months_released_before_first_trading_day_are_available_from_day_one():
    """회귀: 지표 표가 시작되기 전에 이미 발표된 월은 첫 거래일부터 쓸 수 있다(버려지면 평균을 못 구한다)."""
    td = pd.bdate_range("1999-03-10", "1999-06-30")
    months = pd.date_range("1997-01-01", "1999-04-01", freq="MS")
    rel = rg.monthly_release_dates(months, td)
    assert rel[pd.Timestamp("1998-12-01")] == td[0]  # 1999-02 발표 → 표 시작일부터
    assert rel[pd.Timestamp("1999-01-01")] == pd.Timestamp("1999-03-10")
    assert rel[pd.Timestamp("1999-02-01")] == pd.Timestamp("1999-04-01")
    vals = pd.Series(4.0, index=months)
    vals[pd.Timestamp("1999-01-01")] = 6.0
    assert rg.monthly_above_avg(vals, td, 12).iloc[0]  # 첫날부터 12개월 평균 비교 가능


def test_unrate_needs_n_months_available():
    td = pd.bdate_range("2020-01-01", "2020-06-30")
    months = pd.date_range("2019-10-01", periods=5, freq="MS")
    vals = pd.Series([3, 3, 3, 3, 9.0], index=months)
    assert not rg.monthly_above_avg(vals, td, 12).any()  # 12개 미만 → 조건 아님(공격)
    assert rg.monthly_above_avg(vals, td, 4).loc["2020-04-01":].any()


def test_lagged_daily_uses_previous_day_only():
    td = pd.bdate_range("2021-01-04", periods=5)
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=td)
    out = rg.lagged_daily(s, td, 1)
    assert pd.isna(out.iloc[0]) and out.iloc[1] == 1.0 and out.iloc[4] == 4.0


# ── 봉인 ─────────────────────────────────────────────────────────────────


def test_seal_rejects_2022_rows():
    idx = pd.to_datetime(["2021-12-30", "2021-12-31", "2022-01-03"])
    df = pd.DataFrame({"close": [1, 2, 3]}, index=idx)
    with pytest.raises(rg.SealError):
        rg.enforce_seal(df)
    with pytest.raises(rg.SealError):
        rg.enforce_seal({"2021-12-31": 1.0, "2022-01-01": 2.0})
    cut = rg.truncate_to_seal(df)
    assert rg.enforce_seal(cut) is cut and cut.index[-1] == pd.Timestamp("2021-12-31")


def test_seal_portfolio_data_truncates_every_series():
    td = pd.bdate_range("2021-12-27", "2022-01-05")
    price = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=td)
    div = pd.Series([0.1], index=[pd.Timestamp("2022-01-04")])
    data = pf.PortfolioData(
        qqq_df=price, qqq_dividends=div, core_df=price, core_dividends=div, qld_df=price, qld_dividends=div,
        reserve_daily_rate=pd.Series(0.0, index=td), fx_by_date={d.date().isoformat(): 1200.0 for d in td},
        fx_fallback_stats={}, dtb3_stats={}, qld_synthesis_check={}, sma_source_df=price,
    )
    sealed = pf.seal_portfolio_data(data, rg.LAST_ALLOWED_DATE)
    for obj in (sealed.qqq_df, sealed.core_df, sealed.qld_df, sealed.reserve_daily_rate, sealed.sma_source_df, sealed.qqq_dividends):
        rg.enforce_seal(obj)
    assert max(sealed.fx_by_date) == "2021-12-31"


# ── 합성 QLD ─────────────────────────────────────────────────────────────


def test_synthetic_qld_leverage_expense_and_borrow():
    td = pd.bdate_range("2005-01-03", periods=4)
    qqq = pd.Series([100.0, 101.0, 99.99, 100.0], index=td)
    borrow = sa.annual_yield_pct_to_daily_rate(pd.Series(3.6, index=td))  # 3.6%/360 = 0.0001
    out = sa.synthesize_qld_daily_returns(qqq, borrow, annual_expense_pct=0.95, leverage=2.0)
    expected = qqq.pct_change().dropna() * 2 - 0.0095 / 252 - 0.0001
    pd.testing.assert_series_equal(out, expected, check_names=False)
    assert abs(out.iloc[0] - (0.02 - 0.0095 / 252 - 0.0001)) < 1e-12


def test_target_weights_rules():
    td = pd.bdate_range("2021-01-04", periods=4)
    above = pd.Series([True, False, False, float("nan")], index=td, dtype=object)
    up = pd.Series([True, False, True, True], index=td)
    l2 = rg.target_weights("L2", above)
    assert list(l2["regime"]) == ["공격", "방어", "방어", "공격"]
    l3 = rg.target_weights("L3", above, up)
    assert list(l3["regime"]) == ["공격", "공격", "방어", "공격"]  # 200일선 아래 AND 실업률 상승일 때만 방어
    l4 = rg.target_weights("L4", above, up)
    assert l4.iloc[0][["core", "qld", "reserve"]].tolist() == [0.5, 0.5, 0.0]
    assert l4.iloc[2][["core", "qld", "reserve"]].tolist() == [0.0, 0.0, 1.0]
    assert (rg.target_weights("L1", above)["qld"] == 1.0).all()


# ── 목표 비중 시뮬레이터 ───────────────────────────────────────────────────


def _cfg():
    return {"backtest": {"total_krw": 40_000_000, "costs": {"commission_buy_pct": 0.07, "commission_sell_pct": 0.07, "fx_spread_pct": 0.1},
                         "tax": {"capital_gains_deduction_krw": 2_500_000, "capital_gains_rate_pct": 22, "dividend_withholding_pct": 15}}}


def _toy_data(td, core_close, qld_close, rate=0.0001):
    def ohlc(c):
        return pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=td)
    return pf.PortfolioData(
        qqq_df=ohlc(core_close), qqq_dividends=pd.Series(dtype=float), core_df=ohlc(core_close), core_dividends=pd.Series(dtype=float),
        qld_df=ohlc(qld_close), qld_dividends=pd.Series(dtype=float), reserve_daily_rate=pd.Series(rate, index=td),
        fx_by_date={d.date().isoformat(): 1000.0 for d in td}, fx_fallback_stats={}, dtb3_stats={}, qld_synthesis_check={},
        sma_source_df=ohlc(core_close),
    )


def test_simulator_switches_next_day_and_counts_switches():
    td = pd.bdate_range("2021-03-01", periods=6)
    core = pd.Series([100, 100, 100, 100, 100, 100.0], index=td)
    qld = pd.Series([50, 55, 60, 30, 30, 30.0], index=td)
    data = _toy_data(td, core, qld, rate=0.0)
    w = rg.target_weights("L2", pd.Series([True, True, False, False, True, True], index=td, dtype=object))
    res = pf.simulate_target_weights(data, _cfg(), td[0].date(), td[-1].date(), w)
    regimes = [r["regime"] for r in res.equity_rows]
    # td[2] 종가에 방어 신호 → td[3] 시가에 국채로. td[4] 종가 공격 → td[5] 시가 복귀
    assert regimes == ["공격", "공격", "공격", "방어", "방어", "공격"]
    assert res.trade_count == 2
    # td[3] 시가(=30)에 QLD를 판다 → 60에서 30으로 떨어진 손실은 피하지 못한다(다음 날 체결)
    assert res.equity_rows[3]["qld_usd"] == 0 and res.equity_rows[3]["reserve_usd"] > 0


def test_simulator_l0_like_hold_matches_buy_and_hold_value():
    td = pd.bdate_range("2021-03-01", periods=4)
    core = pd.Series([100, 110, 121, 133.1], index=td)
    data = _toy_data(td, core, core, rate=0.0)
    w = rg.target_weights("L0", pd.Series(True, index=td, dtype=object))
    res = pf.simulate_target_weights(data, _cfg(), td[0].date(), td[-1].date(), w)
    v = [r["total_krw"] for r in res.equity_rows]
    assert res.trade_count == 0
    assert v[-1] / v[0] == pytest.approx(1.331, rel=1e-6)


def test_may_tax_paid_from_reserve_when_no_risky_assets():
    td = pd.bdate_range("2020-04-01", "2021-05-31")
    core = pd.Series(np.linspace(100, 200, len(td)), index=td)
    data = _toy_data(td, core, core, rate=0.0)
    above = pd.Series(True, index=td, dtype=object)
    above.loc["2020-12-01":] = False  # 2020년에 이익 실현 후 방어(국채)로
    w = rg.target_weights("L2", above)
    res = pf.simulate_target_weights(data, _cfg(), td[0].date(), td[-1].date(), w)
    paid = [t for t in res.broker.tax_log if t["year"] == 2020]
    assert paid and paid[0]["tax_krw"] > 0
    assert res.broker.qld.shares == 0 and res.broker.core.shares == 0
    # 국채에서 꺼내 냈다 — 남는 음수는 꺼낼 때 수수료(0.07%)만큼(P6 settle_may_tax와 같은 처리)
    usd_paid = paid[0]["tax_krw"] / (1000.0 * (1 - 0.001))
    assert -usd_paid * 0.0008 < res.broker.cash_usd <= 0.01
    _ = date
