"""core/sw1.py·scripts/sw1_experiments.py 테스트 (네트워크 없음, 합성 데이터)."""

import numpy as np
import pandas as pd
import pytest

from core import sw1
from scripts import sw1_data_check as dc
from scripts import sw1_experiments as sx


def _days(n=40):
    return pd.bdate_range("2021-01-04", periods=n)


def _px(days, open_=None, close=None):
    o = pd.Series(100.0, index=days) if open_ is None else open_
    c = pd.Series(100.0, index=days) if close is None else close
    return pd.DataFrame({"open": o, "close": c})


def test_buy_next_day_open_and_sell_after_hold():
    days = _days()
    close = pd.Series(np.arange(len(days), dtype=float) + 100, index=days)
    opens = close - 0.5
    sig = pd.DataFrame({"ticker": ["A"], "d0": [days[3]]})
    tr, sk = sw1.make_trades(sig, {"A": _px(days, opens, close)}, days, hold=5)
    r = tr.iloc[0]
    assert r["buy_day"] == days[4] and r["buy_open"] == opens[days[4]]
    assert r["sell_day"] == days[9] and r["sell_close"] == close[days[9]] and r["exit_reason"] == "만기"
    assert sk.empty


def test_seal_boundary_excluded():
    days = _days(10)
    sig = pd.DataFrame({"ticker": ["A"], "d0": [days[5]]})
    tr, sk = sw1.make_trades(sig, {"A": _px(days)}, days, hold=5)  # 매수 6 + 5 = 11 > 마지막 9
    assert tr.empty and sk.iloc[0]["skip"] == "봉인 경계"
    tr, sk = sw1.make_trades(sig, {"A": _px(days)}, days, hold=3)  # 6 + 3 = 9 → 가능
    assert len(tr) == 1


def test_missing_price_and_delisting():
    days = _days()
    sig = pd.DataFrame({"ticker": ["A", "B"], "d0": [days[3], days[3]]})
    a = _px(days[10:])  # 매수일 행 없음
    b = _px(days[:7])   # 매수(4) 뒤 6일째 상장폐지
    tr, sk = sw1.make_trades(sig, {"A": a, "B": b}, days, hold=5)
    assert sk.set_index("ticker").at["A", "skip"] == "가격 공백"
    r = tr.set_index("ticker").loc["B"]
    assert r["sell_day"] == days[6] and r["exit_reason"] == "가격 끝"


def test_ignore_new_signal_while_holding():
    days = _days()
    sig = pd.DataFrame({"ticker": ["A", "A", "A"], "d0": [days[2], days[5], days[8]]})
    tr, sk = sw1.make_trades(sig, {"A": _px(days)}, days, hold=5)  # 보유 3~8
    assert list(tr["d0"]) == [days[2], days[8]]  # 매도일과 같은 날 반응일은 무시하지 않음
    assert list(sk["skip"]) == ["보유 중"]


def test_stop_loss_uses_close_from_next_day():
    days = _days()
    close = pd.Series(100.0, index=days)
    close[days[4]] = 91.0   # 매수일 종가는 손절 판단에 안 씀
    close[days[6]] = 91.9
    tr, _ = sw1.make_trades(pd.DataFrame({"ticker": ["A"], "d0": [days[3]]}), {"A": _px(days, None, close)}, days, 10, stop_pct=8.0)
    assert tr.iloc[0]["sell_day"] == days[6] and tr.iloc[0]["exit_reason"] == "손절"


def test_eps_version_buys_after_both_dates():
    days = _days()
    sig = pd.DataFrame({"ticker": ["A", "B"], "d0": [days[3], days[10]], "q": [days[6], days[2]]})
    tr, _ = sw1.make_trades(sig, {"A": _px(days), "B": _px(days)}, days, 5, buy_after_col="q")
    t = tr.set_index("ticker")
    assert t.at["A", "buy_day"] == days[7]   # 10-Q 다음 날
    assert t.at["B", "buy_day"] == days[11]  # 10-Q가 먼저 나왔으면 반응일 다음 날 (반응일 전 매수 금지)


def test_krw_after_cost():
    r = sw1.krw_after_cost(0.0, 1000.0, 1000.0)
    assert r == pytest.approx((1 - 0.001) ** 2 * (1 - 0.0007) / 1.0007 - 1)
    assert sw1.krw_after_cost(0.1, 1000.0, 1100.0) > 0.2  # 환율 이익 포함


def test_ledger_tax_profit_proportional_no_deduction():
    ret = np.array([0.10, 0.30, -0.20, 0.05, -0.10])
    yr = np.array([2012, 2012, 2012, 2013, 2013])
    t = sw1.ledger_tax(ret, yr)
    # 2012: 순이익 0.2 → 세금 0.044, 이익 0.1:0.3 비례. 2013: 순손실 → 0
    assert t == pytest.approx([0.011, 0.033, 0.0, 0.0, 0.0])
    many = sw1.ledger_tax_many(np.vstack([ret, ret[::-1]]), np.vstack([yr - 2012, yr[::-1] - 2012]), 3)
    assert many[0] == pytest.approx(t)
    assert many[1] == pytest.approx(sw1.ledger_tax(ret[::-1], yr[::-1]))


def test_percentile_and_verdict():
    assert sw1.percentile_of(5.0, np.array([1, 2, 5, 9])) == pytest.approx(62.5)
    ex = pd.Series([0.01] * 300 + [-0.001] * 10)
    yrs = pd.Series(list(range(2012, 2022)) * 31)
    v = sw1.verdict(ex, yrs, 80.0)
    assert v["pass"] is True and v["pos_years"] == 10
    assert sw1.verdict(ex[:100], yrs[:100], 80.0)["pass"] is None  # 300건 미만 → 판정 불가
    assert sw1.verdict(ex, yrs, 74.9)["pass"] is False


def test_no_lookahead_signal_ignores_prices_after_reaction_day():
    """반응일 뒤 가격·거래량을 아무렇게 바꿔도 신호 판정·매수일이 그대로여야 한다."""
    days = _days(60)
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(days))), index=days)
    vol = pd.Series(rng.uniform(900, 1100, len(days)), index=days)
    d0 = days[30]
    close[d0:] *= 1.08
    vol[d0] = 5000
    bench = pd.Series(300.0, index=days)
    px = pd.DataFrame({"close": close, "volume": vol})
    base = dc.signal_on(px, bench, d0)
    px2 = px.copy()
    px2.loc[px2.index > d0, "close"] *= rng.uniform(0.1, 10, (px2.index > d0).sum())
    px2.loc[px2.index > d0, "volume"] = 0
    bench2 = bench.copy()
    bench2[bench2.index > d0] = 1.0
    assert dc.signal_on(px2, bench2, d0) == base and base["signal"]
    # 잘라낸 데이터(반응일까지)와 전체 데이터의 판정이 같다
    assert dc.signal_on(px.loc[:d0], bench.loc[:d0], d0) == base
    # 매수일은 항상 반응일 뒤
    ohlc = pd.DataFrame({"open": close, "close": close})
    tr, _ = sw1.make_trades(pd.DataFrame({"ticker": ["A"], "d0": [d0]}), {"A": ohlc}, days, 10)
    assert tr.iloc[0]["buy_day"] > d0


def test_lf_sha256_ignores_crlf():
    assert sx.lf_sha256(b"a\r\nb\n") == sx.lf_sha256(b"a\nb\n")


def test_close_mismatch():
    i = pd.bdate_range("2021-01-04", periods=3)
    assert sx.close_mismatch(pd.Series([1.0, 2.0, 3.0], i), pd.Series([1.0, 2.0, 3.003], i)) == pytest.approx(1 - 3 / 3.003)
    assert sx.close_mismatch(pd.Series([1.0], i[:1]), pd.Series([1.0], i[1:2])) == float("inf")
