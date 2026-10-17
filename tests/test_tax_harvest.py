"""core/tax_harvest.py 테스트 (합성 데이터, 네트워크·시트 없음) — 12월 공제 안내."""

import pandas as pd
import pytest

from core import tax_harvest as th


def _fills(rows):
    return pd.DataFrame(rows, columns=["date", "ticker", "side", "qty", "price", "fx_rate"])


def test_moving_average_cost_and_realized_by_year():
    f = _fills([
        ("2025-03-03", "QQQM", "buy", 10, 200.0, 1300.0),   # 2,600,000
        ("2025-06-02", "QQQM", "buy", 10, 220.0, 1400.0),   # 3,080,000 → 평균 284,000/주
        ("2025-09-01", "QQQM", "sell", 5, 230.0, 1350.0),   # 5 × 310,500 − 5 × 284,000 = 132,500
        ("2026-02-02", "QQQM", "sell", 5, 240.0, 1400.0),   # 5 × 336,000 − 5 × 284,000 = 260,000
        ("2026-03-02", "NVDA", "buy", 1, 100.0, 1400.0),    # 다른 종목은 무시
    ])
    p = th.moving_average_position(f)
    assert p.qty == 10 and p.avg_cost_krw == pytest.approx(284_000)
    assert p.realized_by_year == {2025: pytest.approx(132_500), 2026: pytest.approx(260_000)}


def test_blank_sell_fx_uses_that_days_close():
    f = _fills([("2026-01-05", "QQQM", "buy", 10, 200.0, 1300.0),
                ("2026-07-01", "QQQM", "sell", 2, 250.0, None)])
    fx = pd.Series([1380.0, 1400.0], index=pd.to_datetime(["2026-06-30", "2026-07-01"]))
    p = th.moving_average_position(f, fx)
    assert p.realized_by_year[2026] == pytest.approx(2 * 250 * 1400 - 2 * 260_000)


def test_plan_subtracts_this_years_realized_gain():
    f = _fills([("2025-01-02", "QQQM", "buy", 100, 100.0, 1000.0),       # 주당 100,000원
                ("2026-05-01", "QQQM", "sell", 10, 150.0, 1000.0)])     # 실현 500,000
    plan = th.harvest_plan(th.moving_average_position(f), 2026, prev_close_usd=160.0, prev_fx=1000.0)
    assert plan.status == "ok" and plan.remaining_krw == pytest.approx(2_000_000)
    assert plan.per_share_gain_krw == pytest.approx(60_000)
    assert plan.sell_qty == 33 and plan.expected_gain_krw == pytest.approx(1_980_000)


def test_loss_state_and_no_room():
    f = _fills([("2025-01-02", "QQQM", "buy", 10, 200.0, 1400.0)])
    loss = th.harvest_plan(th.moving_average_position(f), 2026, prev_close_usd=180.0, prev_fx=1400.0)
    assert loss.status == "loss" and "올해는 팔 필요 없음(손실 상태)" in th.build_lines(loss, pd.Timestamp("2026-12-29"))
    f2 = _fills([("2025-01-02", "QQQM", "buy", 100, 100.0, 1000.0),
                 ("2026-05-01", "QQQM", "sell", 50, 160.0, 1000.0)])   # 실현 3,000,000 > 250만
    full = th.harvest_plan(th.moving_average_position(f2), 2026, prev_close_usd=160.0, prev_fx=1000.0)
    assert full.status == "no_room" and full.remaining_krw == 0 and full.sell_qty == 0


def test_no_record_gives_method_only():
    plan = th.harvest_plan(th.moving_average_position(_fills([])), 2026, 250.0, 1400.0)
    assert plan.status == "no_record"
    lines = th.build_lines(plan, pd.Timestamp("2026-12-29"))
    assert any("체결 탭에 평균 매입가 한 줄을 적으면 수량을 계산해 드려요" in x for x in lines)
    assert not any("주," in x for x in lines)  # 수량 없음
    text = "\n".join(lines)
    assert "팔고 같은 날 같은 수량 다시 사기, 달러로 매매(환전 안 함)" in text and "12/29까지 체결" in text
    assert all(ord(ch) < 0x1F000 for ch in text)


def _nyse_like(year, holidays):
    days = pd.bdate_range(f"{year}-11-20", f"{year}-12-31")
    return days[~days.isin(pd.to_datetime(holidays))]


def test_deadline_two_trading_days_before_last_with_year_end_holidays():
    d26 = _nyse_like(2026, ["2026-11-26", "2026-12-25"])  # 마지막 12/31(목) → 12/29(화)
    assert th.deadline(d26, 2026) == pd.Timestamp("2026-12-29")
    d22 = _nyse_like(2022, ["2022-11-24", "2022-12-26"])  # 마지막 12/30(금), 12/26 휴장 → 12/28(수)
    assert th.deadline(d22, 2022) == pd.Timestamp("2022-12-28")
    d24 = _nyse_like(2024, ["2024-11-28", "2024-12-25"])  # 마지막 12/31(화), 12/25 휴장 → 12/27(금)
    assert th.deadline(d24, 2024) == pd.Timestamp("2024-12-27")


def test_show_days_first_december_day_and_first_on_or_after_15th():
    d = _nyse_like(2026, ["2026-11-26", "2026-12-25"])
    assert th.show_days(d, 2026) == [pd.Timestamp("2026-12-01"), pd.Timestamp("2026-12-15")]
    d27 = _nyse_like(2027, ["2027-11-25", "2027-12-24"])  # 2027-12-15는 수요일
    assert th.show_days(d27, 2027) == [pd.Timestamp("2027-12-01"), pd.Timestamp("2027-12-15")]
    d29 = _nyse_like(2029, ["2029-11-22", "2029-12-25"])  # 2029-12-15는 토요일 → 12/17(월)
    assert th.show_days(d29, 2029)[1] == pd.Timestamp("2029-12-17")
