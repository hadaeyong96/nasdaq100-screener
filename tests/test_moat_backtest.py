"""core/moat_backtest.py 순수 함수 테스트 (H4 과거 검증 부품)."""

from __future__ import annotations

import random

import pytest

from core.moat_backtest import (
    PeriodResult,
    allocate_equal_weight_with_sector_cap,
    annualized_return,
    chain_periods_with_tax,
    compute_stock_period_return,
    draw_random_portfolio,
    percentile_rank,
    portfolio_period_return,
)


def test_equal_weight_no_cap_needed():
    sectors = {"A": "S1", "B": "S2", "C": "S3"}
    weights = allocate_equal_weight_with_sector_cap(["A", "B", "C"], sectors, 50)
    assert weights == pytest.approx({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})


def test_sector_cap_redistributes_excess():
    # 4종목, 두 개가 같은 업종 S1 -> 동일비중이면 S1=50%, 캡 30%를 넘음
    sectors = {"A": "S1", "B": "S1", "C": "S2", "D": "S3"}
    weights = allocate_equal_weight_with_sector_cap(["A", "B", "C", "D"], sectors, 30)
    sector_totals: dict[str, float] = {}
    for t, w in weights.items():
        sector_totals[sectors[t]] = sector_totals.get(sectors[t], 0.0) + w
    assert sector_totals["S1"] <= 0.30 + 1e-6
    assert sum(weights.values()) == pytest.approx(1.0)
    # 초과분은 비례 배분되어 S2, S3가 원래(0.25)보다 커야 함
    assert weights["C"] > 0.25
    assert weights["D"] > 0.25


def test_sector_cap_impossible_still_sums_to_one_and_does_not_hang():
    # 모든 종목이 한 업종 -> 캡을 지킬 수 없음. 무한루프 없이 끝나고 합은 1
    sectors = {"A": "S1", "B": "S1", "C": "S1"}
    weights = allocate_equal_weight_with_sector_cap(["A", "B", "C"], sectors, 30)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_allocate_empty():
    assert allocate_equal_weight_with_sector_cap([], {}, 30) == {}


def test_draw_random_portfolio_deterministic_with_seed():
    universe = [f"T{i}" for i in range(20)]
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    picked1 = draw_random_portfolio(universe, 5, rng1)
    picked2 = draw_random_portfolio(universe, 5, rng2)
    assert picked1 == picked2
    assert len(picked1) == 5
    assert set(picked1).issubset(set(universe))


def test_draw_random_portfolio_caps_at_universe_size():
    universe = ["A", "B", "C"]
    rng = random.Random(1)
    picked = draw_random_portfolio(universe, 10, rng)
    assert sorted(picked) == sorted(universe)


def test_compute_stock_period_return_basic():
    costs_cfg = {"slippage_pct": 0.05, "commission_buy_pct": 0.07, "commission_sell_pct": 0.07}
    price_factor, div_yield = compute_stock_period_return(100.0, 110.0, 2.0, costs_cfg, 0.15)
    buy_price = 100.0 * 1.0005 * 1.0007
    sell_price = 110.0 * 0.9995 * 0.9993
    assert price_factor == pytest.approx(sell_price / buy_price)
    assert div_yield == pytest.approx((2.0 / buy_price) * 0.85)


def test_compute_stock_period_return_apply_sell_costs_false_is_pure_mark():
    # 계획서 4장 "봉인" — 마지막 평가는 실제 매도가 아니라 그 값 그대로 쓴다(슬리피지·매도수수료 없음)
    costs_cfg = {"slippage_pct": 0.05, "commission_buy_pct": 0.07, "commission_sell_pct": 0.07}
    price_factor, div_yield = compute_stock_period_return(100.0, 110.0, 0.0, costs_cfg, 0.15, apply_sell_costs=False)
    buy_price = 100.0 * 1.0005 * 1.0007
    assert price_factor == pytest.approx(110.0 / buy_price)


def test_compute_stock_period_return_zero_entry_price_is_neutral():
    costs_cfg = {"slippage_pct": 0.05, "commission_buy_pct": 0.07, "commission_sell_pct": 0.07}
    price_factor, div_yield = compute_stock_period_return(0.0, 110.0, 2.0, costs_cfg, 0.15)
    assert (price_factor, div_yield) == (1.0, 0.0)


def test_portfolio_period_return_weighted_average():
    stock_returns = {"A": (1.2, 0.01), "B": (0.8, 0.02)}
    weights = {"A": 0.5, "B": 0.5}
    price_factor, div_yield = portfolio_period_return(stock_returns, weights)
    assert price_factor == pytest.approx(1.0)
    assert div_yield == pytest.approx(0.015)


def test_portfolio_period_return_missing_stock_defaults_neutral():
    # 시세를 못 구한 종목은 (1.0, 0.0)으로 취급 (현금처럼)
    weights = {"A": 0.5, "MISSING": 0.5}
    price_factor, div_yield = portfolio_period_return({"A": (1.5, 0.0)}, weights)
    assert price_factor == pytest.approx(1.25)
    assert div_yield == pytest.approx(0.0)


def test_chain_periods_with_tax_single_period_pays_tax_on_gain():
    tax_cfg = {"capital_gains_deduction_krw": 2_500_000, "capital_gains_rate_pct": 22}
    periods = [
        PeriodResult(
            entry_date="2019-04-01", exit_date="2020-04-01",
            fx_entry=1200.0, fx_exit=1250.0, price_return_factor=1.5, dividend_aftertax_yield=0.0
        )
    ]
    rows = chain_periods_with_tax(periods, starting_capital_usd=10_000.0, tax_cfg=tax_cfg)
    assert len(rows) == 2
    assert rows[0]["date"] == "2019-04-01"
    assert rows[0]["total_krw"] == pytest.approx(10_000.0 * 1200.0)
    assert rows[1]["date"] == "2020-04-01"
    # 실현손익(원화) = 10000 * 0.5 * 1250 = 6,250,000 -> 과세대상 = 6,250,000-2,500,000=3,750,000 * 22% = 825,000
    expected_gain_krw = 10_000.0 * 0.5 * 1250.0
    expected_tax_krw = (expected_gain_krw - 2_500_000) * 0.22
    expected_gross_krw = 10_000.0 * 1.5 * 1250.0
    assert rows[1]["total_krw"] == pytest.approx(expected_gross_krw - expected_tax_krw)
    assert rows[1]["tax_krw"] == pytest.approx(expected_tax_krw)


def test_chain_periods_with_tax_loss_pays_no_tax():
    tax_cfg = {"capital_gains_deduction_krw": 2_500_000, "capital_gains_rate_pct": 22}
    periods = [
        PeriodResult(
            entry_date="2019-04-01", exit_date="2020-04-01",
            fx_entry=1200.0, fx_exit=1200.0, price_return_factor=0.8, dividend_aftertax_yield=0.0
        )
    ]
    rows = chain_periods_with_tax(periods, starting_capital_usd=10_000.0, tax_cfg=tax_cfg)
    assert rows[1]["tax_krw"] == pytest.approx(0.0)
    assert rows[1]["total_krw"] == pytest.approx(10_000.0 * 0.8 * 1200.0)


def test_chain_periods_with_tax_empty_returns_empty():
    assert chain_periods_with_tax([], 10_000.0, {"capital_gains_deduction_krw": 0, "capital_gains_rate_pct": 22}) == []


def test_chain_periods_compounds_across_two_periods():
    tax_cfg = {"capital_gains_deduction_krw": 0, "capital_gains_rate_pct": 0}
    periods = [
        PeriodResult("2019-04-01", "2020-04-01", 1200.0, 1200.0, 1.10, 0.0),
        PeriodResult("2020-04-01", "2021-04-01", 1200.0, 1200.0, 1.10, 0.0),
    ]
    rows = chain_periods_with_tax(periods, 10_000.0, tax_cfg)
    assert rows[-1]["date"] == "2021-04-01"
    assert rows[-1]["total_krw"] == pytest.approx(10_000.0 * 1.10 * 1.10 * 1200.0)


def test_percentile_rank_basic():
    dist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile_rank(5, dist) == 50.0
    assert percentile_rank(10, dist) == 100.0
    assert percentile_rank(0, dist) == 0.0


def test_percentile_rank_empty_distribution_defaults_to_50():
    assert percentile_rank(5, []) == 50.0


def test_annualized_return_basic():
    # 2배가 되는데 1년 걸리면 100%
    assert annualized_return(100.0, 200.0, 1.0) == pytest.approx(100.0)


def test_annualized_return_invalid_inputs_return_none():
    assert annualized_return(0.0, 100.0, 1.0) is None
    assert annualized_return(100.0, 0.0, 1.0) is None
    assert annualized_return(100.0, 200.0, 0.0) is None
