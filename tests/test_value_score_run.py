"""scripts/value_score_zone3_run — 가치평가 장부(원화 계좌·세금·갈아타기·수시 매도). 합성 시장만, 네트워크 없음."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from data.market_calendar import trading_days_between
from scripts import value_score_zone3_run as run

DAYS = trading_days_between(date(2012, 4, 2), date(2014, 12, 31))
FX = 1000.0


def _mes():
    s = pd.Series(DAYS, index=DAYS)
    return [pd.Timestamp("2012-03-30")] + list(s.groupby([DAYS.year, DAYS.month]).max())


def _market(prices: dict[str, pd.Series]) -> run.Market:
    cl = dict(prices)
    cl[run.QQQ] = pd.Series(100.0, index=DAYS)
    dv = {t: pd.Series(0.0, index=DAYS) for t in cl}
    return run.Market.from_frames(DAYS, cl, dv, np.full(len(DAYS), FX), _mes(), cl[run.QQQ])


def _board(scores: dict[str, dict], sic2: dict[str, str]) -> run.Board:
    """scores: {티커: {월말 Timestamp 또는 'default': 점수}}"""
    rows = []
    for me in _mes():
        for t, sc in scores.items():
            v = sc.get(me, sc.get("default"))
            rows.append({"date": me, "ticker": t, "universe": True, "total": v, "peg_value": 1.0})
    df = pd.DataFrame(rows)
    return run.Board(df, df["total"], sic2)


def test_buy_sell_costs_and_krw_gain():
    m = _market({"A": pd.Series(10.0, index=DAYS)})
    b = run.Book(m, run.FULL)
    b.cash = 1000.0
    j = m.ti["A"]
    b.buy(0, j, 1000.0)
    q = 1000.0 / (10 * 1.001 * 1.0007)
    assert b.hold[j][0] == pytest.approx(q)
    m.ffill[5, j] = 12.0
    usd = b.sell(5, j, q)
    assert usd == pytest.approx(q * 12 * 0.999 * 0.9993)
    assert b.realized[2012] == pytest.approx(usd * FX - 1000.0 * FX)
    assert j not in b.hold


def test_tax_paid_first_trading_day_of_may():
    m = _market({"A": pd.Series(10.0, index=DAYS)})
    b = run.Book(m, run.FULL)
    b.realized[2012] = 12_500_000.0
    b.close_year(m.di[pd.Timestamp("2012-12-31")])
    assert b.pending_tax == (2012, pytest.approx(2_200_000.0))
    b.cash = 5000.0
    may = m.di[pd.Timestamp("2013-05-01")]
    assert may in m.may_first
    b.pay_tax(may)
    assert b.cash == pytest.approx(5000.0 - 2_200_000.0 / (FX * 0.999))
    assert b.pending_tax is None


def test_harvest_uses_up_deduction_and_keeps_position():
    m = _market({"A": pd.Series(10.0, index=DAYS)})
    b = run.Book(m, run.FULL)
    j = m.ti["A"]
    b.hold[j] = [10_000.0, 10_000.0 * 5.0 * FX, 0]  # 주당 취득 5달러 → 큰 평가 이익
    b.cash = 1_000.0
    b.realized[2012] = 1_000_000.0
    b.harvest(10)
    assert b.realized[2012] == pytest.approx(2_500_000.0)
    assert b.hold[j][0] == pytest.approx(10_000.0, rel=1e-6)  # 판 수량만큼 다시 삼


def test_xirr_simple():
    t0 = pd.Timestamp("2012-01-01")
    assert run.xirr([(t0, -100.0), (t0 + pd.Timedelta(days=365.25 * 2), 121.0)]) == pytest.approx(10.0, abs=1e-6)


def test_quarter_rule_needs_two_quarters_and_one_year():
    """B는 처음부터 35점 미만 — 2번 연속은 2012-09에 되지만 보유 1년 전이라 못 팔고, 1년 지난 뒤 판정에서 판다."""
    prices = {"A": pd.Series(10.0, index=DAYS), "B": pd.Series(20.0, index=DAYS)}
    m = _market(prices)
    board = _board({"A": {"default": 60.0}, "B": {pd.Timestamp("2012-03-30"): 50.0, "default": 30.0}},
                   {"A": "10", "B": "20"})
    r = run.run_value(m, board, run.FULL, record=True)
    sells = [(str(m.days[i].date()), m.tickers[j]) for (i, k, j, *_rest) in r["book"].log if k == "sell" and _rest[2].startswith("분기")]
    # 2013-03-28 분기 말 판정 → 다음 거래일 2013-04-01은 첫 매수(2012-04-02)부터 364일 → 못 팖. 2013-06-28 → 2013-07-01 매도
    assert sells == [("2013-07-01", "B")]


def test_initial_pick_threshold_and_sector_cap():
    prices = {t: pd.Series(10.0, index=DAYS) for t in "ABCDE"}
    m = _market(prices)
    board = _board({"A": {"default": 70.0}, "B": {"default": 69.0}, "C": {"default": 68.0}, "D": {"default": 67.0},
                    "E": {"default": 48.0}}, {"A": "10", "B": "10", "C": "10", "D": "10", "E": "20"})
    r = run.run_value(m, board, run.FULL, record=True)
    first = sorted(m.tickers[j] for (i, k, j, *_r) in r["book"].log if i == 0 and k == "buy")
    assert first == ["A", "B", "C"]  # 업종 10 최대 3, E는 49점 미만
    assert r["stats"]["init"] == 3


def test_december_swap_partial_by_deduction():
    """2012-11 말: 보유 A 40점, 비보유 C 60점(같은 업종) → 12월 실행일에 A를 공제 한도만큼(최대 절반) 팔아 C를 산다."""
    idx = DAYS
    pa = pd.Series(10.0, index=idx)
    pa[idx >= pd.Timestamp("2012-06-01")] = 20.0  # A 주가 두 배 → 1주당 이익 큼
    prices = {"A": pa, "C": pd.Series(10.0, index=idx)}
    m = _market(prices)
    nov = pd.Timestamp("2012-11-30")
    board = _board({"A": {"default": 50.0, nov: 40.0}, "C": {"default": 10.0, nov: 60.0}}, {"A": "10", "C": "10"})
    r = run.run_value(m, board, run.FULL, record=True)
    b = r["book"]
    swaps = r["stats"]["swaps"]
    assert swaps and swaps[0][1:3] == ("A", "C")
    dx = m.days[[i for i, y in m.dec_exec.items() if y == 2012][0]]
    assert swaps[0][0] == str(dx.date()) == "2012-12-27"
    # 남은 공제 250만 원 ÷ 1주당 이익 — 보유 수량의 절반보다 작다
    q0 = 100_000_000 * 0.999 / FX / 10 / (10 * 1.001 * 1.0007)  # 시작 1/10 몫 (A는 납입으로 더 샀을 수 있음 → 비율만 본다)
    assert 0 < swaps[0][3] < 0.5
    assert b.realized[2012] == pytest.approx(2_500_000.0, rel=1e-6)  # 갈아타기 + 공제 소진으로 한도까지
    assert q0 > 0


def test_delisted_sold_at_last_close():
    pa = pd.Series(10.0, index=DAYS)
    pb = pd.Series(30.0, index=DAYS)
    pb[DAYS > pd.Timestamp("2013-01-15")] = np.nan
    m = _market({"A": pa, "B": pb})
    board = _board({"A": {"default": 60.0}, "B": {"default": 60.0}}, {"A": "10", "B": "20"})
    r = run.run_value(m, board, run.FULL, record=True)
    s = [(str(m.days[i].date()), usd / q) for (i, k, j, q, usd, why) in r["book"].log if k == "sell" and m.tickers[j] == "B"]
    assert s[0][0] == "2013-01-16"
    assert s[0][1] == pytest.approx(30.0 * 0.999 * 0.9993)


def test_contribution_goes_to_cash_when_vacancy():
    m = _market({"A": pd.Series(10.0, index=DAYS)})
    board = _board({"A": {"default": 60.0}}, {"A": "10"})
    r = run.run_value(m, board, run.FULL, record=True)
    buys_a = [i for (i, k, j, *_r) in r["book"].log if k == "buy" and m.tickers[j] == "A" and _r[2] == "납입"]
    assert buys_a == []  # 보유 1종목 < 10 → 빈자리 → 납입은 현금


def test_harvest_full_resell_keeps_first_buy_date():
    """[B1] 평가 이익이 공제보다 작아 전량을 팔았다 다시 사도 첫 매수일(1년 조건·분기 표시)이 바뀌지 않는다."""
    m = _market({"A": pd.Series(10.0, index=DAYS)})
    b = run.Book(m, run.FULL)
    j = m.ti["A"]
    b.hold[j] = [100.0, 100.0 * 9.0 * FX, 3]  # 평가 이익 약 10만 원 < 공제 250만 원 → 전량 매도 후 다시 삼
    b.cash = 10.0
    b.harvest(200)
    assert j in b.hold and b.hold[j][2] == 3
    assert b.hold[j][0] == pytest.approx(100.0, rel=0.01)


def test_harvest_runs_even_without_costs_and_tax():
    """[B2] 비용 0·세금 없음 판에서도 12월 공제 소진(규칙)을 한다 — 단계 비교가 같은 규칙이 되게."""
    pa = pd.Series(10.0, index=DAYS)
    pa[DAYS >= pd.Timestamp("2012-06-01")] = 20.0
    m = _market({"A": pa})
    board = _board({"A": {"default": 60.0}}, {"A": "10"})
    r = run.run_value(m, board, run.ZERO, record=True)
    assert r["stats"]["harvest_trades"] > 0
    assert r["book"].realized[2012] == pytest.approx(2_500_000.0, rel=1e-6)
