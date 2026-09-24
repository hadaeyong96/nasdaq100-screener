"""core/tax.py 양도세·배당 원천징수 테스트 (P5-1 6번: 공제 250만 원, 손실 연도). 네트워크 없음."""

from __future__ import annotations

from core import tax


def test_capital_gains_tax_zero_when_under_deduction():
    assert tax.capital_gains_tax(2_000_000) == 0.0


def test_capital_gains_tax_exactly_at_deduction_boundary_is_zero():
    assert tax.capital_gains_tax(2_500_000) == 0.0


def test_capital_gains_tax_applies_rate_above_deduction():
    # 500만 - 250만 공제 = 250만 과세표준 * 22% = 55만
    assert tax.capital_gains_tax(5_000_000) == 550_000.0


def test_capital_gains_tax_zero_in_loss_year():
    assert tax.capital_gains_tax(-3_000_000) == 0.0


def test_capital_gains_tax_custom_deduction_and_rate():
    assert tax.capital_gains_tax(10_000_000, deduction_krw=1_000_000, rate=0.20) == 1_800_000.0


def test_dividend_after_withholding_default_rate():
    assert tax.dividend_after_withholding(100_000) == 85_000.0


def test_dividend_after_withholding_custom_rate():
    assert tax.dividend_after_withholding(100_000, rate=0.10) == 90_000.0
