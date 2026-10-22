"""전략 2개 더(갈아타기·2배 균형) 순수 함수 — 네트워크 없음."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import two_more as tm


def test_par_bond_and_bond_returns():
    assert tm.par_bond_price(0.05, 0.05) == pytest.approx(1.0)
    assert tm.par_bond_price(0.06, 0.05) < 1.0 < tm.par_bond_price(0.04, 0.05)
    d = pd.bdate_range("2000-01-03", periods=5)
    r = tm.bond_returns(pd.Series([5.0] * 5, index=d))
    assert r.iloc[0] == 0 and r.iloc[1:].tolist() == pytest.approx([0.05 / 252] * 4)
    up = tm.bond_returns(pd.Series([5.0, 6.0], index=d[:2]))
    assert up.iloc[1] == pytest.approx(tm.par_bond_price(0.06, 0.05) - 1 + 0.05 / 252)  # 금리 1%p ↑ → 약 −10%
    assert -0.12 < up.iloc[1] < -0.08


def test_bond_returns_no_future():
    d = pd.bdate_range("2000-01-03", periods=50)
    y = pd.Series(np.linspace(5, 6, 50), index=d)
    assert tm.bond_returns(y).iloc[30] == pytest.approx(tm.bond_returns(y.iloc[:31]).iloc[-1])


def test_splice_real():
    d = pd.bdate_range("2000-01-03", periods=4)
    real = pd.Series([np.nan, 10.0, 11.0, 11.0], index=d)
    px = tm.splice_real(pd.Series([0.0, 0.01, 0.01, 0.01], index=d), real)
    assert px.tolist() == pytest.approx([1.0, 1.01, 1.01 * 1.1, 1.01 * 1.1])  # 실제 첫 수익률(셋째 날)부터


def test_total_return_monthly_dividend():
    d = pd.bdate_range("2000-01-03", "2000-02-29")
    p = pd.Series(100.0, index=d)
    tr = tm.total_return_with_monthly_dividend(p, pd.Series([12.0, 24.0], index=pd.to_datetime(["2000-01-01", "2000-02-01"])))
    jan = (d.month == 1).sum()
    assert tr.iloc[jan - 1] == pytest.approx((1 + 1.0 / jan / 100) ** (jan - 1))  # 첫날은 0, 1월 배당 1을 고르게
    assert tr.iloc[-1] > tr.iloc[jan - 1]


def test_lookback_and_choice_and_positions():
    d = pd.bdate_range("1999-01-01", "2001-03-30")
    n = len(d)
    prices = np.ones((n, 6))
    prices[:, tm.STOCK] = np.linspace(100, 200, n)
    prices[:, tm.BOND] = np.linspace(100, 110, n)
    prices[:, tm.GOLD] = np.linspace(100, 300, n)
    prices[:, tm.CASH] = np.linspace(1, 1.05, n)
    me = tm.month_end_positions(d, 0)
    assert all(d[p].is_month_end or d[p + 1].month != d[p].month for p in me[:-1])
    assert [d[p].month for p in tm.quarter_end_positions(d, 0)][:4] == [3, 6, 9, 12]
    p = me[-1]
    r = tm.lookback_returns(prices, d, p, 12, {tm.GOLD: pd.Timestamp("2000-08-30")})
    assert tm.GOLD not in r and tm.switch_choice(r) == tm.STOCK  # 금은 12개월 전 가격이 없어 후보 아님
    r2 = tm.lookback_returns(prices, d, p, 6, {tm.GOLD: pd.Timestamp("2000-08-30")})
    assert tm.switch_choice(r2) == tm.GOLD
    assert tm.switch_choice({tm.STOCK: -0.1, tm.BOND: -0.05, tm.CASH: 0.03}) == tm.CASH
    # 미래 데이터 방지: p 이후 값을 바꿔도 p 판단은 같다
    prices2 = prices.copy()
    prices2[p + 1:] = 1.0
    assert tm.lookback_returns(prices2, d, me[-3], 12, {}) == tm.lookback_returns(prices, d, me[-3], 12, {})


def test_sharpe():
    nav = np.array([1.0, 1.01, 1.0, 1.02])
    cash = np.ones(4)
    r = nav[1:] / nav[:-1] - 1
    assert tm.sharpe(nav, cash) == pytest.approx(r.mean() * 252 / (r.std(ddof=1) * np.sqrt(252)))
