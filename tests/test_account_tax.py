"""core/account_tax.py 테스트 — 손으로 계산한 예제 포함, 네트워크 없음."""

from __future__ import annotations

import pytest

from core import account_tax as at


# ── 해외계좌 양도세: 250만 원 공제 경계 (3개) ────────────────────────────────


def test_capital_gains_tax_exactly_at_deduction_is_zero():
    assert at.capital_gains_tax(2_500_000, 2_500_000, 0.22) == 0.0


def test_capital_gains_tax_just_above_deduction():
    # 과세대상 1원 × 22% = 0.22원
    assert at.capital_gains_tax(2_500_001, 2_500_000, 0.22) == pytest.approx(0.22)


def test_capital_gains_tax_well_above_deduction():
    # 과세대상 750만원 × 22% = 165만원
    assert at.capital_gains_tax(10_000_000, 2_500_000, 0.22) == pytest.approx(1_650_000)


# ── ISA 해지 세금: 200만 원(일반형) 경계 (3개) ────────────────────────────────


def test_isa_exit_tax_exactly_at_exemption_is_zero():
    assert at.isa_exit_tax(2_000_000, 2_000_000, 9.9) == 0.0


def test_isa_exit_tax_just_above_exemption():
    assert at.isa_exit_tax(2_000_001, 2_000_000, 9.9) == pytest.approx(0.099)


def test_isa_exit_tax_well_above_exemption():
    # 과세대상 300만원 × 9.9% = 29.7만원
    assert at.isa_exit_tax(5_000_000, 2_000_000, 9.9) == pytest.approx(297_000)


def test_isa_exit_tax_loss_is_zero():
    assert at.isa_exit_tax(-1_000_000, 2_000_000, 9.9) == 0.0


# ── 연금 세액공제: 600만 원 한도 경계 (3개) ──────────────────────────────────


def test_pension_tax_credit_exactly_at_cap():
    assert at.pension_tax_credit(6_000_000, 6_000_000, 16.5) == pytest.approx(990_000)


def test_pension_tax_credit_above_cap_is_capped():
    # 700만원 납입해도 한도(600만원) 초과분은 공제 안 됨 — 600만원 때와 동일
    assert at.pension_tax_credit(7_000_000, 6_000_000, 16.5) == pytest.approx(990_000)


def test_pension_tax_credit_below_cap():
    assert at.pension_tax_credit(3_000_000, 6_000_000, 16.5) == pytest.approx(495_000)


def test_pension_tax_credit_lower_rate():
    assert at.pension_tax_credit(6_000_000, 6_000_000, 13.2) == pytest.approx(792_000)


# ── 연금 인출 세금 ────────────────────────────────────────────────────────────


def test_pension_withdrawal_tax_pension_rate():
    assert at.pension_withdrawal_tax(100_000_000, 5.5) == pytest.approx(5_500_000)


def test_pension_withdrawal_tax_lump_sum_rate():
    assert at.pension_withdrawal_tax(100_000_000, 16.5) == pytest.approx(16_500_000)


# ── ISA 한도 ─────────────────────────────────────────────────────────────────


def test_isa_annual_room_basic():
    assert at.isa_annual_room(12_000_000, 20_000_000) == pytest.approx(8_000_000)


def test_isa_annual_room_exhausted_is_zero_not_negative():
    assert at.isa_annual_room(25_000_000, 20_000_000) == 0.0


def test_isa_lifetime_room_basic():
    assert at.isa_lifetime_room(60_000_000, 100_000_000) == pytest.approx(40_000_000)


# ── 이동평균법 원장 ──────────────────────────────────────────────────────────


def test_moving_average_ledger_buy_then_sell_all_realizes_total_gain():
    ledger = at.MovingAverageLedger()
    ledger.buy(1000.0, 100.0)  # 10주 @100
    assert ledger.shares == pytest.approx(10.0)
    assert ledger.avg_cost_per_share == pytest.approx(100.0)
    proceeds, gain = ledger.sell(1500.0, 150.0)  # 10주 @150 전량 매도
    assert proceeds == pytest.approx(1500.0)
    assert gain == pytest.approx(500.0)  # (150-100)*10
    assert ledger.shares == pytest.approx(0.0)


def test_moving_average_ledger_blends_cost_across_two_buys():
    ledger = at.MovingAverageLedger()
    ledger.buy(1000.0, 100.0)  # 10주 @100
    ledger.buy(1000.0, 200.0)  # 5주 @200
    # 평균원가 = (1000+1000) / (10+5) = 133.33
    assert ledger.avg_cost_per_share == pytest.approx(2000.0 / 15.0)


def test_moving_average_ledger_partial_sell_keeps_avg_cost():
    ledger = at.MovingAverageLedger()
    ledger.buy(1000.0, 100.0)  # 10주 @100
    proceeds, gain = ledger.sell(500.0, 150.0)  # 약 3.33주 매도 @150
    shares_sold = 500.0 / 150.0
    assert gain == pytest.approx(shares_sold * (150.0 - 100.0))
    assert ledger.avg_cost_per_share == pytest.approx(100.0)  # 남은 주식 평균원가는 그대로


def test_moving_average_shares_for_target_gain():
    ledger = at.MovingAverageLedger()
    ledger.buy(1000.0, 100.0)  # 10주 @100, 평균원가 100
    # 현재가 150 -> 주당 이익 50. 목표 250 -> 5주
    shares = ledger.shares_for_target_gain(150.0, 250.0)
    assert shares == pytest.approx(5.0)


def test_moving_average_shares_for_target_gain_zero_when_no_gain():
    ledger = at.MovingAverageLedger()
    ledger.buy(1000.0, 100.0)
    assert ledger.shares_for_target_gain(90.0, 250.0) == 0.0  # 손실 중 -> 못 채움


def test_moving_average_shares_for_target_gain_capped_at_holdings():
    ledger = at.MovingAverageLedger()
    ledger.buy(1000.0, 100.0)  # 10주
    shares = ledger.shares_for_target_gain(1000.0, 1_000_000.0)  # 터무니없이 큰 목표
    assert shares == pytest.approx(10.0)  # 보유 전량을 넘지 않음


# ── 선입선출법 원장 ──────────────────────────────────────────────────────────


def test_fifo_ledger_sells_oldest_lot_first():
    ledger = at.FifoLedger()
    ledger.buy(1000.0, 100.0)  # lot1: 10주 @100
    ledger.buy(1000.0, 200.0)  # lot2: 5주 @200
    # 12주 매도(가격 250) -> lot1 전부(10주, 이익 (250-100)*10=1500) + lot2 2주(이익 (250-200)*2=100)
    proceeds, gain = ledger.sell(12 * 250.0, 250.0)
    assert gain == pytest.approx(1500.0 + 100.0)
    assert ledger.shares == pytest.approx(3.0)  # lot2 남은 3주


def test_fifo_ledger_total_gain_matches_moving_average_on_full_liquidation():
    # 전량 매도 시 실현손익 합계는 방식과 무관하게 같아야 한다(계획서 3장 "증명").
    fifo = at.FifoLedger()
    avg = at.MovingAverageLedger()
    for amount, price in [(1000.0, 100.0), (2000.0, 150.0), (500.0, 80.0)]:
        fifo.buy(amount, price)
        avg.buy(amount, price)
    final_price = 120.0
    _, fifo_gain = fifo.sell(fifo.shares * final_price, final_price)
    _, avg_gain = avg.sell(avg.shares * final_price, final_price)
    assert fifo_gain == pytest.approx(avg_gain)


def test_fifo_shares_for_target_gain_within_one_lot():
    ledger = at.FifoLedger()
    ledger.buy(1000.0, 100.0)  # lot1: 10주 @100
    # 현재가 150, 목표 250 -> lot1 안에서 5주(이익 50/주 * 5 = 250)
    shares = ledger.shares_for_target_gain(150.0, 250.0)
    assert shares == pytest.approx(5.0)
    # 미리보기라 실제 lot은 안 바뀜
    assert ledger.shares == pytest.approx(10.0)


def test_fifo_shares_for_target_gain_crosses_lot_boundary():
    ledger = at.FifoLedger()
    ledger.buy(500.0, 100.0)  # lot1: 5주 @100
    ledger.buy(500.0, 50.0)   # lot2: 10주 @50
    price = 150.0
    # lot1 전부: 5주 * (150-100)=250 이익. 목표 400 -> lot1(250) + lot2에서 150/((150-50))=1.5주
    shares = ledger.shares_for_target_gain(price, 400.0)
    assert shares == pytest.approx(5.0 + 1.5)


def test_fifo_shares_for_target_gain_skips_loss_lots():
    ledger = at.FifoLedger()
    ledger.buy(1000.0, 200.0)  # lot1: 5주 @200 (손실 중)
    ledger.buy(500.0, 50.0)    # lot2: 10주 @50 (이익 중)
    price = 100.0  # lot1은 손실(200->100), lot2는 이익(50->100)
    shares = ledger.shares_for_target_gain(price, 300.0)  # lot2에서만 채워야 함: 300/50=6주
    assert shares == pytest.approx(6.0)
