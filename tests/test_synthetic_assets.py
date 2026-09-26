"""core.synthetic_assets 테스트 (P6-1). 네트워크 없이 합성 데이터로 검증."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import synthetic_assets as sa


def test_annual_yield_pct_to_daily_rate_uses_360_day_count():
    s = pd.Series([3.6, 7.2], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    out = sa.annual_yield_pct_to_daily_rate(s)
    assert out.iloc[0] == pytest.approx(3.6 / 100 / 360)
    assert out.iloc[1] == pytest.approx(7.2 / 100 / 360)


def test_align_rate_to_trading_days_forward_fills_bond_market_holiday():
    rate_map = {"2020-01-02": 1.5, "2020-01-06": 1.6}  # 2020-01-03(금)은 국채 공시가 없다고 가정
    trading_days = list(pd.bdate_range("2020-01-02", "2020-01-06"))  # 화~월(주말 제외 영업일)
    out = sa.align_rate_to_trading_days(rate_map, trading_days)
    assert out.loc[pd.Timestamp("2020-01-03")] == 1.5  # 직전 값으로 채움
    assert out.loc[pd.Timestamp("2020-01-06")] == 1.6


def test_align_rate_to_trading_days_no_prior_value_is_nan():
    rate_map = {"2020-01-03": 1.5}
    trading_days = list(pd.bdate_range("2020-01-02", "2020-01-03"))
    out = sa.align_rate_to_trading_days(rate_map, trading_days)
    assert pd.isna(out.loc[pd.Timestamp("2020-01-02")])


def test_synthesize_qld_daily_returns_matches_formula():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    qqq_close = pd.Series([100.0, 102.0, 101.0], index=dates)
    borrow = pd.Series([0.0001, 0.0001, 0.0001], index=dates)
    out = sa.synthesize_qld_daily_returns(qqq_close, borrow, annual_expense_pct=0.95, leverage=2.0)
    qqq_ret_day2 = 102.0 / 100.0 - 1
    expected_day2 = qqq_ret_day2 * 2.0 - 0.95 / 100 / 252 - 0.0001
    assert out.iloc[0] == pytest.approx(expected_day2)
    assert len(out) == 2  # 첫날은 전일 종가가 없어 제외


def test_synthesize_qld_daily_returns_default_leverage_and_expense():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
    qqq_close = pd.Series([100.0, 110.0], index=dates)
    borrow = pd.Series([0.0, 0.0], index=dates)
    out = sa.synthesize_qld_daily_returns(qqq_close, borrow)
    assert out.iloc[0] == pytest.approx(0.10 * 2.0 - 0.95 / 100 / 252)


def test_levels_from_daily_returns_compounds():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    returns = pd.Series([0.10, -0.05, 0.0], index=dates)
    out = sa.levels_from_daily_returns(returns, start_level=1.0)
    assert out.iloc[0] == pytest.approx(1.10)
    assert out.iloc[1] == pytest.approx(1.10 * 0.95)
    assert out.iloc[2] == pytest.approx(1.10 * 0.95)


def test_splice_synthetic_returns_before_real_matches_at_boundary():
    pre_dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
    pre_returns = pd.Series([0.01, 0.02], index=pre_dates)  # 2020-01-01->01-02, 01-02->01-03 수익률
    real_dates = pd.to_datetime(["2020-01-03", "2020-01-04"])
    real_df = pd.DataFrame({"open": [50.0, 51.0], "high": [50.0, 51.0], "low": [50.0, 51.0], "close": [50.0, 51.0]}, index=real_dates)

    out = sa.splice_synthetic_returns_before_real(pre_returns, real_df)

    assert list(out.index) == list(pre_dates) + list(real_dates)
    # 접합 경계 직전 날(01-02)의 수준 * (1+그날 수익률) == 접합 경계(01-03) 실제 종가
    boundary_close = 50.0
    day02_level = out.loc[pd.Timestamp("2020-01-02"), "close"]
    assert day02_level * (1 + pre_returns.loc[pd.Timestamp("2020-01-02")]) == pytest.approx(boundary_close)
    # open/high/low가 합성 구간에서는 close와 같다(근사)
    assert out.loc[pd.Timestamp("2020-01-01"), "open"] == out.loc[pd.Timestamp("2020-01-01"), "close"]


def test_splice_synthetic_returns_before_real_empty_pre_returns_is_noop():
    real_dates = pd.to_datetime(["2020-01-03"])
    real_df = pd.DataFrame({"open": [50.0], "high": [50.0], "low": [50.0], "close": [50.0]}, index=real_dates)
    out = sa.splice_synthetic_returns_before_real(pd.Series(dtype=float), real_df)
    assert list(out.index) == list(real_dates)
    assert out.loc[real_dates[0], "close"] == 50.0


def test_no_lookahead_daily_rate_alignment_stable_when_truncated():
    """t일까지 자른 데이터의 t일 값이 전체 데이터의 t일 값과 같아야 한다 (미래 데이터 방지)."""
    rate_map = {"2020-01-02": 1.5, "2020-01-06": 1.6, "2020-01-08": 1.7}
    trading_days_full = list(pd.bdate_range("2020-01-02", "2020-01-08"))
    trading_days_trunc = list(pd.bdate_range("2020-01-02", "2020-01-06"))

    full = sa.align_rate_to_trading_days(rate_map, trading_days_full)
    trunc = sa.align_rate_to_trading_days(rate_map, trading_days_trunc)

    assert full.loc[pd.Timestamp("2020-01-06")] == trunc.loc[pd.Timestamp("2020-01-06")]
