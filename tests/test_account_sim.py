"""core/account_sim.py 테스트 — 네트워크 없이, 합성 데이터로 돈다."""

from __future__ import annotations

from datetime import date

import pytest

from core import account_sim as sim


def _cfg():
    return {
        "costs": {"commission_pct": 0.07, "fx_spread_pct": 0.1},
        "expense_ratio": {"domestic_etf_pct": 0.10},
        "tax": {
            "overseas_account": {"capital_gains_deduction_krw": 2_500_000, "capital_gains_rate_pct": 22, "dividend_withholding_pct": 15},
            "isa": {
                "general_type_exemption_krw": 2_000_000, "underprivileged_type_exemption_krw": 4_000_000,
                "rate_pct": 9.9, "annual_limit_krw": 20_000_000, "lifetime_limit_krw": 100_000_000,
                "mandatory_years": 3, "reopen_cycle_years": 3,
            },
            "pension": {
                "deduction_cap_krw": 6_000_000, "credit_rate_pct": [16.5, 13.2], "refund_paid_month_offset": 5,
                "withdrawal_tax_pct": {"pension": 5.5, "lump_sum": 16.5},
                "fixed_monthly_contribution_krw": 500_000,
            },
            "domestic_wrapper_dividend_haircut_pct": 15,
            "reserve_interest_tax_pct": 15.4,
        },
        "reserve": {"skim_pct_of_monthly_allocation": 20, "drawdown_trigger_pct": -20, "reference_series": "qqq_close"},
    }


def _flat_steps(n_months: int, qqq_price=100.0, core_price=100.0, fx=1000.0) -> list[sim.MonthStep]:
    """가격·환율·배당·보수·이자 전부 변화 없는 합성 월별 시리즈(0% 성장 테스트용)."""
    return _make_steps([qqq_price] * n_months, core_price=core_price, fx=fx)


def _make_steps(qqq_prices: list[float], core_price=None, fx=1000.0) -> list[sim.MonthStep]:
    """월별 종가 목록으로 합성 시리즈를 만든다 — 달력 연도 경계(is_year_start/end)를
    실제로 넘어가며 올바르게 계산한다(여러 해짜리 ISA 재가입 테스트용).

    core_price가 없으면 qqq_prices와 같은 값을 쓴다(해외계좌=QQQM이 QQQ와 같이 움직인다고 봄).
    """
    n = len(qqq_prices)
    steps = []
    for i, qp in enumerate(qqq_prices):
        cp = qp if core_price is None else (core_price[i] if isinstance(core_price, list) else core_price)
        year, month = 2000 + i // 12, i % 12 + 1
        is_year_start = i % 12 == 0
        is_year_end = (i % 12 == 11) or (i == n - 1)
        steps.append(
            sim.MonthStep(
                month_index=i, contribution_date=date(year, month, 1), qqq_price=qp, core_price=cp, fx=fx,
                qqq_div_yield_since_prev=0.0, core_div_per_share_since_prev=0.0,
                domestic_fee_factor_since_prev=1.0, reserve_factor_since_prev=1.0,
                is_year_end=is_year_end, is_year_start=is_year_start, calendar_year=year,
            )
        )
    return steps


# ── 봉인 ─────────────────────────────────────────────────────────────────────


def test_enforce_window_not_sealed_rejects_after_boundary():
    with pytest.raises(sim.SealViolationError):
        sim.enforce_window_not_sealed(date(2022, 1, 1), date(2021, 12, 31))


def test_enforce_window_not_sealed_accepts_boundary_and_before():
    sim.enforce_window_not_sealed(date(2021, 12, 31), date(2021, 12, 31))
    sim.enforce_window_not_sealed(date(2015, 1, 1), date(2021, 12, 31))


# ── 납입 배분(ISA 한도 초과분 해외계좌로) ─────────────────────────────────────


def test_allocate_isa_then_overseas_within_room():
    isa, overseas = sim.allocate_isa_then_overseas_krw(1_000_000, 20_000_000, 100_000_000)
    assert isa == pytest.approx(1_000_000)
    assert overseas == pytest.approx(0.0)


def test_allocate_isa_then_overseas_exceeds_annual_room():
    # 남은 연 한도가 50만 원뿐인데 100만 원을 넣으려 함 -> 50만은 ISA, 50만은 해외계좌
    isa, overseas = sim.allocate_isa_then_overseas_krw(1_000_000, 500_000, 100_000_000)
    assert isa == pytest.approx(500_000)
    assert overseas == pytest.approx(500_000)


def test_allocate_isa_then_overseas_exceeds_lifetime_room():
    isa, overseas = sim.allocate_isa_then_overseas_krw(1_000_000, 20_000_000, 300_000)
    assert isa == pytest.approx(300_000)
    assert overseas == pytest.approx(700_000)


def test_allocate_isa_then_overseas_room_exhausted():
    isa, overseas = sim.allocate_isa_then_overseas_krw(1_000_000, 0.0, 100_000_000)
    assert isa == 0.0
    assert overseas == pytest.approx(1_000_000)


def test_net_of_commission_basic():
    assert sim.net_of_commission(1000.0, 0.07) == pytest.approx(1000.0 / 1.0007)


# ── 0% 수익률: K0·K2 세후자산 = 납입총액 − 비용, 세금 0 ───────────────────────


def test_k0_zero_growth_zero_tax_and_value_close_to_contributed():
    cfg = _cfg()
    steps = _flat_steps(4)  # 3개월 납입 + 1개월 평가
    result = sim.run_candidate("K0", steps, start_capital_krw=40_000_000, saving_krw=1_000_000, cfg=cfg)
    total_contributed = 40_000_000 + 1_000_000 * 3
    assert result.total_tax_krw == pytest.approx(0.0, abs=1e-6)
    assert result.final_posttax_ex_pension_krw < total_contributed  # 비용만큼 적음
    assert result.final_posttax_ex_pension_krw > total_contributed * 0.99  # 비용은 1% 미만


def test_k2_zero_growth_zero_tax_and_value_close_to_contributed():
    cfg = _cfg()
    steps = _flat_steps(4)
    result = sim.run_candidate("K2", steps, start_capital_krw=40_000_000, saving_krw=1_000_000, cfg=cfg)
    total_contributed = 40_000_000 + 1_000_000 * 3
    assert result.total_tax_krw == pytest.approx(0.0, abs=1e-6)
    assert result.final_posttax_ex_pension_krw < total_contributed
    assert result.final_posttax_ex_pension_krw > total_contributed * 0.99


def test_k0_zero_growth_trade_count_matches_months():
    cfg = _cfg()
    steps = _flat_steps(4)
    result = sim.run_candidate("K0", steps, start_capital_krw=40_000_000, saving_krw=1_000_000, cfg=cfg)
    # 시작매수 1 + 3개월 매수 3 + 최종 매도 1 = 5
    assert result.trade_count == 5


# ── 미래 데이터 방지: build_month_steps를 t일까지 자른 결과와 전체 결과가 같음 ──


def test_build_month_steps_truncation_matches_full_up_to_cutoff():
    trading_days = [date(2020, 1, d) for d in (2, 3, 6, 7, 8)] + [date(2020, 2, d) for d in (3, 4, 5)] + [date(2020, 3, 2)]
    qqq_close = {d: 100.0 + i for i, d in enumerate(trading_days)}
    core_close = dict(qqq_close)
    qqq_div = {date(2020, 1, 7): 0.5}
    core_div = {date(2020, 1, 7): 0.4}
    fx = {d: 1200.0 for d in trading_days}
    reserve_rate = {d: 0.0001 for d in trading_days}

    full = sim.build_month_steps(
        trading_days, qqq_close, core_close, qqq_div, core_div, fx, reserve_rate,
        window_start=date(2020, 1, 1), window_end=date(2020, 3, 2), domestic_fee_pct=0.10,
    )
    cut = sim.build_month_steps(
        trading_days, qqq_close, core_close, qqq_div, core_div, fx, reserve_rate,
        window_start=date(2020, 1, 1), window_end=date(2020, 2, 5), domestic_fee_pct=0.10,
    )
    # cut은 1월·2월 두 달만 있어야 하고, 그 두 달의 값은 full의 같은 두 달과 같아야 한다
    assert [s.contribution_date for s in cut] == [s.contribution_date for s in full[:2]]
    for a, b in zip(cut, full[:2]):
        assert a.qqq_price == b.qqq_price
        assert a.qqq_div_yield_since_prev == pytest.approx(b.qqq_div_yield_since_prev)
        assert a.reserve_factor_since_prev == pytest.approx(b.reserve_factor_since_prev)


# ── K4 급락 신호 ─────────────────────────────────────────────────────────────


def test_compute_drawdown_events_triggers_at_20_percent_drop_and_recovers():
    days = [date(2020, 1, i) for i in range(1, 11)]
    # 100(고점) -> 하락 -> 79(=-21%, 트리거) -> 회복 -> 101(신고점, 회복)
    prices = {days[0]: 100, days[1]: 100, days[2]: 95, days[3]: 85, days[4]: 79,
              days[5]: 80, days[6]: 90, days[7]: 95, days[8]: 100, days[9]: 101}
    events = sim.compute_drawdown_events(days, prices, trigger_pct=-20)
    assert len(events) == 2
    assert events[0].kind == "trigger" and events[0].date == days[4]
    assert events[1].kind == "recovered" and events[1].date == days[8]  # 종가 100이 직전 고점(100)과 같아지는 날


def test_compute_drawdown_events_no_trigger_when_drop_insufficient():
    days = [date(2020, 1, i) for i in range(1, 5)]
    prices = {days[0]: 100, days[1]: 95, days[2]: 90, days[3]: 85}  # -15%, 기준(-20%) 미달
    events = sim.compute_drawdown_events(days, prices, trigger_pct=-20)
    assert events == []


def test_skimming_phase_by_month_reflects_trigger_and_recovery():
    days = [date(2020, 1, i) for i in range(1, 11)]
    prices = {days[0]: 100, days[1]: 100, days[2]: 95, days[3]: 85, days[4]: 79,
              days[5]: 80, days[6]: 90, days[7]: 95, days[8]: 100, days[9]: 101}
    events = sim.compute_drawdown_events(days, prices, trigger_pct=-20)
    month_dates = [days[0], days[4], days[6], days[9]]
    phases = sim.skimming_phase_by_month(month_dates, events)
    assert phases == [True, False, False, True]  # 트리거 당일부터 비활성, 회복일부터 다시 활성


# ── K4 적립 분배 ─────────────────────────────────────────────────────────────


def test_split_equity_and_reserve_active():
    equity, reserve = sim.split_equity_and_reserve(1_000_000, 20, skimming_active=True)
    assert equity == pytest.approx(800_000)
    assert reserve == pytest.approx(200_000)


def test_split_equity_and_reserve_paused_goes_all_to_equity():
    equity, reserve = sim.split_equity_and_reserve(1_000_000, 20, skimming_active=False)
    assert equity == pytest.approx(1_000_000)
    assert reserve == 0.0


# ── 불변식 테스트 (사용자 지시 2026-10-01, ISA 재가입 취득원가 버그 점검) ──────────


def test_k2_flat_price_zero_cost_multi_cycle_pays_zero_tax_ever():
    """회귀 테스트(핵심 버그): 가격이 전혀 안 오르고 비용도 0이면, ISA를 몇 번을
    재가입하든(이 테스트는 2번 재가입 — 84개월) 세금이 단 한 번도 나오면 안 된다.
    버그가 있으면(재가입 때 이전 원금을 다시 이익으로 잡으면) 2번째 재가입부터 세금이
    생긴다 — 가격이 하나도 안 올랐는데도.
    """
    cfg = _cfg()
    cfg["costs"] = {"commission_pct": 0.0, "fx_spread_pct": 0.0}
    cfg["expense_ratio"] = {"domestic_etf_pct": 0.0}
    steps = _flat_steps(85)  # 84개월 납입(재가입 2번: 월 36, 72) + 1개월 평가
    result = sim.run_candidate("K2", steps, start_capital_krw=0, saving_krw=1_000_000, cfg=cfg)
    assert result.total_tax_krw == pytest.approx(0.0, abs=1e-6)
    assert result.final_posttax_ex_pension_krw == pytest.approx(84_000_000.0, rel=1e-9)


def test_k2_reinvestment_overflow_goes_to_overseas_with_correct_cost_basis():
    """ISA 해지 후 재예치 금액이 그 해 연 한도를 넘으면, 넘는 돈은 해외계좌로 가고
    그 취득가는 "옮긴 금액"(해지 후 세후 금액 중 넘는 부분)이어야 한다 — 0이거나
    원래 ISA 원가여서는 안 된다(사용자 지시 2026-10-01 진단 2번).

    연 한도(2천만 원)보다 훨씬 큰 월 저축(1천만 원 × 12 = 1.2억/년)으로 ISA가 금방
    한도를 넘게 만들어, 36개월째 재가입 시점에 반드시 해외계좌로 넘어가게 한다.
    """
    cfg = _cfg()
    cfg["costs"] = {"commission_pct": 0.0, "fx_spread_pct": 0.0}
    steps = _flat_steps(37, qqq_price=100.0, core_price=100.0, fx=1000.0)  # 36개월 납입 + 평가 1개월
    result = sim.run_candidate("K2", steps, start_capital_krw=0, saving_krw=10_000_000, cfg=cfg)
    # 월 저축이 연 한도를 훨씬 넘으므로 애초에 매달도 해외계좌로 넘침 + 재가입 때도 넘침.
    # 옮긴 돈이 전부 취득가 0으로 잡혔다면(버그) 최종 양도세가 비정상적으로 커진다 —
    # 가격이 안 올랐으므로(성장 없음) 올바른 구현이면 해외계좌 양도세도 0이어야 한다.
    assert result.total_tax_krw == pytest.approx(0.0, abs=1.0)


def test_zero_tax_rates_k0_and_k2_differ_only_by_fee_and_commission():
    cfg = _cfg()
    cfg["tax"]["overseas_account"]["capital_gains_rate_pct"] = 0.0
    cfg["tax"]["overseas_account"]["dividend_withholding_pct"] = 0.0
    cfg["tax"]["isa"]["rate_pct"] = 0.0
    cfg["tax"]["reserve_interest_tax_pct"] = 0.0
    prices = [100.0 * (1.01 ** i) for i in range(13)]  # 완만히 상승(재가입 없음, 12개월)
    steps = _make_steps(prices)
    k0 = sim.run_candidate("K0", steps, start_capital_krw=40_000_000, saving_krw=1_000_000, cfg=cfg)
    k2 = sim.run_candidate("K2", steps, start_capital_krw=40_000_000, saving_krw=1_000_000, cfg=cfg)
    assert k0.total_tax_krw == pytest.approx(0.0, abs=1.0)
    assert k2.total_tax_krw == pytest.approx(0.0, abs=1.0)
    # 세금이 전부 0이면 둘의 차이는 보수(0.10%대 0.15%, 무시할 수준)·수수료 차이뿐 —
    # 12개월짜리 짧은 구간이라 그 차이는 전체 자산의 1% 안쪽이어야 한다.
    diff_pct = abs(k0.final_posttax_ex_pension_krw - k2.final_posttax_ex_pension_krw) / k0.final_posttax_ex_pension_krw * 100
    assert diff_pct < 1.0


def test_total_tax_never_exceeds_22pct_of_pretax_profit():
    prices = [100.0 * (1.02 ** i) for i in range(85)]  # 꾸준히 상승, 84개월 납입 + 평가(재가입 2번)
    steps = _make_steps(prices)
    cfg = _cfg()
    result = sim.run_candidate("K2", steps, start_capital_krw=40_000_000, saving_krw=1_000_000, cfg=cfg)
    total_contributed = 40_000_000 + 1_000_000 * 84
    pretax_profit = result.final_posttax_ex_pension_krw + result.total_tax_krw - total_contributed
    assert result.total_tax_krw <= max(pretax_profit, 0.0) * 0.22 + 1.0


def test_k1_not_materially_worse_than_k0_when_harvesting_only_realizes_gains():
    prices = [100.0 * (1.02 ** i) for i in range(25)]  # 24개월 납입(연말 수확 2번) + 평가
    steps = _make_steps(prices)
    cfg = _cfg()
    k0 = sim.run_candidate("K0", steps, start_capital_krw=40_000_000, saving_krw=1_000_000, cfg=cfg)
    k1 = sim.run_candidate("K1", steps, start_capital_krw=40_000_000, saving_krw=1_000_000, cfg=cfg, cost_basis="moving_average")
    # 수확은 이익이 있을 때만 "팔고 바로 재매수"라 왕복 수수료 정도만 손해 — 포트폴리오
    # 전체 대비 크지 않아야 한다(느슨하게 0.5%).
    assert k1.final_posttax_ex_pension_krw >= k0.final_posttax_ex_pension_krw * 0.995
