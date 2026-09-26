"""engine.portfolio 테스트 (P6-1). 네트워크 없이 합성 데이터로 검증."""

from __future__ import annotations

import pandas as pd
import pytest

from engine import portfolio as pf

CFG = {
    "backtest": {
        "total_krw": 40_000_000,
        "costs": {"commission_buy_pct": 0.07, "commission_sell_pct": 0.07, "fx_spread_pct": 0.1},
        "tax": {
            "capital_gains_deduction_krw": 2_500_000, "capital_gains_rate_pct": 22,
            "dividend_withholding_pct": 15, "payment_month": 5,
        },
    }
}


def _flat_df(dates, prices) -> pd.DataFrame:
    idx = pd.DatetimeIndex(dates)
    return pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices}, index=idx)


def _mk_data(dates: list[str], qqq_prices: list[float], fx_rate: float = 1300.0) -> pf.PortfolioData:
    df = _flat_df(dates, qqq_prices)
    empty_div = pd.Series(dtype=float)
    fx_by_date = {d: fx_rate for d in dates}
    reserve_rate = pd.Series(0.0, index=df.index)  # 국채 수익 0(테스트 단순화)
    return pf.PortfolioData(
        qqq_df=df, qqq_dividends=empty_div, core_df=df.copy(), core_dividends=empty_div,
        qld_df=df.copy(), qld_dividends=empty_div, reserve_daily_rate=reserve_rate,
        fx_by_date=fx_by_date, fx_fallback_stats={}, dtb3_stats={}, qld_synthesis_check={},
    )


# ── PortfolioAsset (이동평균법) ──────────────────────────────────────────────


def test_portfolio_asset_buy_then_partial_sell_average_cost():
    asset = pf.PortfolioAsset()
    asset.buy(1000.0, 100.0, commission_pct=0.0, apply_costs=False)  # 10주 @100
    asset.buy(1000.0, 200.0, commission_pct=0.0, apply_costs=False)  # 5주 @200, 평단 (1000+1000)/15=133.33
    avg_cost = asset.cost_usd / asset.shares
    assert avg_cost == pytest.approx(2000 / 15)

    proceeds, gain = asset.sell(500.0, 250.0, commission_pct=0.0, apply_costs=False)  # 2주 매도 @250
    shares_sold = 500.0 / 250.0
    expected_gain = 500.0 - avg_cost * shares_sold
    assert gain == pytest.approx(expected_gain)
    assert proceeds == pytest.approx(500.0)
    assert asset.shares == pytest.approx(15 - shares_sold)


def test_portfolio_asset_sell_applies_commission():
    asset = pf.PortfolioAsset()
    asset.buy(1000.0, 100.0, commission_pct=0.0, apply_costs=False)
    proceeds, _ = asset.sell(500.0, 100.0, commission_pct=1.0, apply_costs=True)
    assert proceeds == pytest.approx(500.0 * 0.99)


# ── P0: 100% 보유 ────────────────────────────────────────────────────────────


def test_p0_buy_and_hold_has_zero_rebalance_trades():
    dates = pd.bdate_range("2020-01-02", "2020-06-30").strftime("%Y-%m-%d").tolist()
    prices = [100.0 + i * 0.1 for i in range(len(dates))]  # 완만한 상승, 위기 없음
    data = _mk_data(dates, prices)
    result = pf.simulate_portfolio(data, CFG, pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date(), "P0")
    assert result.trade_count == 0


def test_p0_dividends_are_reinvested_same_day_without_counting_as_a_trade():
    dates = pd.bdate_range("2020-01-02", "2020-06-30").strftime("%Y-%m-%d").tolist()
    prices = [100.0] * len(dates)  # 가격 고정 — 배당 재투자 효과만 분리해서 본다
    data = _mk_data(dates, prices)
    import dataclasses

    div_date = pd.Timestamp(dates[10])
    data = dataclasses.replace(data, core_dividends=pd.Series([1.0], index=[div_date]))  # 주당 $1 배당

    result = pf.simulate_portfolio(data, CFG, pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date(), "P0")

    assert result.trade_count == 0  # 배당 재투자는 "매매 횟수"에 안 잡힌다
    shares_before_div = (CFG["backtest"]["total_krw"] / 1300.0 * (1 - 0.001)) / 100.0
    assert result.broker.core.shares > shares_before_div  # 배당만큼 주식 수가 늘어남


def test_p0_tax_only_charged_at_final_liquidation_not_annually():
    dates = pd.bdate_range("2019-01-02", "2020-12-31").strftime("%Y-%m-%d").tolist()
    prices = [100.0 * (1 + 0.0005) ** i for i in range(len(dates))]  # 꾸준한 상승
    data = _mk_data(dates, prices)
    end = pd.Timestamp(dates[-1]).date()
    result = pf.simulate_portfolio(data, CFG, pd.Timestamp(dates[0]).date(), end, "P0")

    # 중간에 5월 정산이 있어도(2020-05) 실현손익이 0이라 세금은 0이어야 한다(매도 자체가 없음)
    assert all(row["tax_krw"] == 0.0 for row in result.broker.tax_log)
    assert result.broker.core.shares > 0  # 청산 전이라 여전히 보유 중

    posttax_a = pf.compute_posttax_a(result, data, CFG, end)
    assert posttax_a["tax_paid_krw"] > 0  # 청산 시점에는 그동안 쌓인 평가차익에 세금이 붙는다


# ── P3: 코어·대기(낙폭 투입 + 회복 재조정) ───────────────────────────────────


def test_p3_deploys_reserve_on_drawdown_and_rebalances_on_recovery():
    dates = pd.bdate_range("2020-01-02", "2020-08-31").strftime("%Y-%m-%d").tolist()
    n = len(dates)
    peak_i = n // 4
    trough_i = n // 2
    prices = []
    for i in range(n):
        if i <= peak_i:
            prices.append(100.0 * (1 + 0.001) ** i)
        elif i <= trough_i:
            frac = (i - peak_i) / (trough_i - peak_i)
            prices.append(prices[peak_i] * (1 - 0.35 * frac))  # -35%까지 하락(−20·−30 발동, −40 미도달)
        else:
            frac = (i - trough_i) / (n - 1 - trough_i)
            prices.append(prices[trough_i] + (prices[peak_i] * 1.05 - prices[trough_i]) * frac)  # 직전 고점 이상으로 회복

    data = _mk_data(dates, prices)
    result = pf.simulate_portfolio(data, CFG, pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date(), "P3")

    assert result.trade_count > 0  # 낙폭 투입 + 회복 재조정 거래가 있어야 한다
    last_row = result.equity_rows[-1]
    total_final = last_row["core_usd"] + last_row["qld_usd"] + last_row["reserve_usd"] + last_row["cash_usd"]
    # 회복 후 60:40으로 재조정됐으므로 코어 비중이 대략 60%에 가까워야 한다
    assert last_row["core_usd"] / total_final == pytest.approx(0.6, abs=0.05)


def test_p3_no_drawdown_never_touches_reserve():
    dates = pd.bdate_range("2020-01-02", "2020-03-31").strftime("%Y-%m-%d").tolist()
    prices = [100.0 * (1 + 0.001) ** i for i in range(len(dates))]  # 계속 상승, 낙폭 없음
    data = _mk_data(dates, prices)
    result = pf.simulate_portfolio(data, CFG, pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date(), "P3")
    assert result.trade_count == 0
    assert result.broker.reserve_usd > 0


# ── P4: 코어·이동평균 스위치 ─────────────────────────────────────────────────


def test_p4_switches_to_defensive_below_sma_and_back_above():
    n = 260
    dates = pd.bdate_range("2019-01-02", periods=n).strftime("%Y-%m-%d").tolist()
    prices = [100.0] * 210 + [90.0] * 20 + [110.0] * (n - 230)  # 200일선 아래로 떨어졌다가 회복
    data = _mk_data(dates, prices)
    result = pf.simulate_portfolio(data, CFG, pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date(), "P4", sma_days=200)
    assert result.trade_count >= 2  # 방어 전환 1회 + 복귀 1회 이상


# ── 지표 함수 ────────────────────────────────────────────────────────────────


def test_compute_longest_recovery_days_finds_max_gap():
    rows = [
        {"date": "2020-01-01", "total_krw": 100},
        {"date": "2020-01-02", "total_krw": 90},
        {"date": "2020-01-03", "total_krw": 95},
        {"date": "2020-01-06", "total_krw": 101},  # 3거래일 만에 회복(01-01 대비)
        {"date": "2020-01-07", "total_krw": 80},
        {"date": "2020-01-08", "total_krw": 70},
        {"date": "2020-01-09", "total_krw": 102},  # 3거래일 만에 회복(01-06 대비)
    ]
    out = pf.compute_longest_recovery_days(rows)
    assert out["longest_recovery_trading_days"] == 3
    assert out["open_ended"] is False


def test_compute_longest_recovery_days_open_ended_when_never_recovers():
    rows = [
        {"date": "2020-01-01", "total_krw": 100},
        {"date": "2020-01-02", "total_krw": 50},
        {"date": "2020-01-03", "total_krw": 60},
    ]
    out = pf.compute_longest_recovery_days(rows)
    assert out["open_ended"] is True
    assert out["longest_recovery_trading_days"] == 2


def test_compute_worst_year_month_picks_minimum():
    rows = [
        {"date": "2020-01-31", "total_krw": 100},
        {"date": "2020-06-30", "total_krw": 80},
        {"date": "2020-12-31", "total_krw": 130},
        {"date": "2021-12-31", "total_krw": 90},
    ]
    out = pf.compute_worst_year_month(rows)
    assert out["worst_year"][0] == 2021  # 2020: +30%, 2021: 90/130-1 ≈ -30.8%


# ── 미래 데이터 방지 회귀 테스트 ──────────────────────────────────────────────


def test_no_lookahead_truncated_run_matches_full_run_up_to_cutoff():
    dates = pd.bdate_range("2020-01-02", "2020-08-31").strftime("%Y-%m-%d").tolist()
    n = len(dates)
    prices = []
    for i in range(n):
        if i < n // 2:
            prices.append(100.0 * (1 - 0.002) ** i)
        else:
            prices.append(prices[-1] * (1 + 0.003) ** 1)
    data = _mk_data(dates, prices)

    cutoff = pd.Timestamp(dates[n // 2]).date()
    full = pf.simulate_portfolio(data, CFG, pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date(), "P3")
    truncated = pf.simulate_portfolio(data, CFG, pd.Timestamp(dates[0]).date(), cutoff, "P3")

    full_rows_by_date = {r["date"]: r for r in full.equity_rows if r["date"] <= cutoff.isoformat()}
    trunc_rows_by_date = {r["date"]: r for r in truncated.equity_rows}
    assert full_rows_by_date.keys() == trunc_rows_by_date.keys()
    for d, row in trunc_rows_by_date.items():
        assert row["total_krw"] == pytest.approx(full_rows_by_date[d]["total_krw"])
