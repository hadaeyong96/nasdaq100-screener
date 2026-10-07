"""D1 적립식 매수 규칙·장부 테스트 (네트워크 없음, 합성 데이터)."""

from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import pytest

from core import dca


def _days(start="2000-01-03", periods=900):
    return pd.bdate_range(start, periods=periods)


def _price(td, seed=1, vol=0.02, drift=0.0003):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, len(td)))), index=td)


def _steps(price, fx=None, rate=0.0001, div=None, n=24, rule="first"):
    td = price.index
    months = list(pd.period_range(td[210].to_period("M") + 1, periods=n, freq="M"))
    facts = dca.signal_facts(price)
    dates = dca.buy_dates(td, months, rule)
    val = dca.buy_dates(td, [months[-1] + 1], "first")[0]
    return dca.build_steps(td, price, fx, pd.Series(rate, index=td), div if div is not None else pd.Series(dtype=float), facts, dates, val)


# ── 공정한 비교: 총 납입 같음, 현금보다 많이 사지 않음 ───────────────────────


@pytest.mark.parametrize("krw", [False, True])
def test_same_contribution_and_no_overspend(krw):
    price = _price(_days(periods=1200), vol=0.03)
    fx = pd.Series(1200.0, index=price.index) if krw else None
    steps = _steps(price, fx=fx, n=36)
    base = 300_000.0 if krw else 300.0
    res = {c: dca.simulate(steps, c, base, krw=krw) for c in dca.CANDIDATES}
    assert len({r["contributed"] for r in res.values()}) == 1
    assert res["D0"]["contributed"] == base * 36
    for c, r in res.items():
        assert r["min_cash"] >= -1e-9, c
    # D0은 납입을 다 쓴다(달러 데이터: 현금 = 이자만 남음 → 0)
    if not krw:
        assert res["D0"]["cash_ratio_avg"] == pytest.approx(0.0, abs=1e-12)


def test_cash_cap_limits_big_buys():
    td = _days(periods=800)
    price = pd.Series(np.linspace(200, 100, len(td)), index=td)  # 계속 하락 → D4는 늘 2배를 원함
    steps = _steps(price, rate=0.0, n=6)
    r = dca.simulate(steps, "D4", 300.0)
    assert r["spent_total"] == pytest.approx(300.0 * 6)  # 원하는 600을 못 쓰고 들어온 만큼만
    assert r["min_cash"] >= 0


# ── 규칙 경계값 ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dd,mult", [(0.0, 0.7), (-0.0999, 0.7), (-0.10, 1.5), (-0.1999, 1.5), (-0.20, 2.0), (-0.2999, 2.0), (-0.30, 3.0), (-0.6, 3.0)])
def test_d2_tiers(dd, mult):
    assert dca.d2_multiple(dd) == mult
    assert dca.rule_amount("D2", 100.0, month=1, t=0, dd=dd) == pytest.approx(100.0 * mult)


def test_d1_down_up_zero():
    assert dca.rule_amount("D1", 100.0, month=1, t=0, last_month_ret=-0.0001) == 150.0
    assert dca.rule_amount("D1", 100.0, month=1, t=0, last_month_ret=0.0) == 50.0
    assert dca.rule_amount("D1", 100.0, month=1, t=0, last_month_ret=0.02) == 50.0


def test_d4_below_above_and_equal():
    assert dca.rule_amount("D4", 100.0, month=1, t=0, below=1.0) == 200.0
    assert dca.rule_amount("D4", 100.0, month=1, t=0, below=0.0) == 50.0
    td = _days(periods=5)
    f = dca.signal_facts(pd.Series([1.0, 1.0, 1.0, 0.5, 2.0], index=td), sma_days=3)
    assert np.isnan(f["below"].iloc[1])
    assert list(f["below"].iloc[2:]) == [0.0, 1.0, 0.0]  # 같으면 위(0), 아래면 1


@pytest.mark.parametrize("month,amt", [(11, 150.0), (12, 150.0), (1, 150.0), (4, 150.0), (5, 50.0), (10, 50.0)])
def test_d5_season(month, amt):
    assert dca.rule_amount("D5", 100.0, month=month, t=0) == amt


def test_value_averaging():
    assert dca.rule_amount("D3", 100.0, month=1, t=0, equity_value=0.0) == 100.0
    assert dca.rule_amount("D3", 100.0, month=1, t=2, equity_value=250.0) == 50.0
    assert dca.rule_amount("D3", 100.0, month=1, t=2, equity_value=320.0) == 0.0  # 넘으면 안 삼(팔지 않음)
    # 가격이 그대로면 D3 = D0 (수수료 때문에 평가액이 조금 모자라 그만큼 더 사려 하지만 현금 한도)
    td = _days(periods=800)
    flat = pd.Series(100.0, index=td)
    steps = _steps(flat, rate=0.0, n=12)
    d3 = dca.simulate(steps, "D3", 300.0)
    d0 = dca.simulate(steps, "D0", 300.0)
    assert d3["final"] == pytest.approx(d0["final"], rel=1e-9)


def test_last_month_ret_definition():
    td = pd.bdate_range("2001-01-01", "2001-04-30")
    close = pd.Series(100.0, index=td)
    close.loc["2001-02"] = 90.0  # 2월 말 90, 1월 말 100 → 3월에 쓰는 지난달 수익 −10%
    close.loc["2001-03"] = 99.0
    f = dca.signal_facts(close, sma_days=3)
    assert f.loc["2001-03-01", "last_month_ret"] == pytest.approx(-0.10)
    assert f.loc["2001-04-02", "last_month_ret"] == pytest.approx(0.10)
    assert np.isnan(f.loc["2001-01-02", "last_month_ret"])


def test_buy_dates_rules():
    td = pd.bdate_range("2021-05-01", "2021-05-31")
    m = [pd.Period("2021-05", "M")]
    assert dca.buy_dates(td, m, "first")[0] == pd.Timestamp("2021-05-03")
    assert dca.buy_dates(td, m, "mid")[0] == pd.Timestamp("2021-05-17")  # 15일은 토요일
    assert dca.buy_dates(td, m, "last")[0] == pd.Timestamp("2021-05-31")


# ── 미래 데이터 방지 ───────────────────────────────────────────────────────


@pytest.mark.parametrize("cut", [250, 500, 800])
def test_signal_facts_truncated_equals_full(cut):
    price = _price(_days(), vol=0.03)
    t = price.index[cut]
    pd.testing.assert_frame_equal(dca.signal_facts(price).loc[:t], dca.signal_facts(price.loc[:t]))


def test_no_negative_shift():
    from scripts import d1_experiments

    for mod in (dca, d1_experiments):
        assert not re.search(r"shift\(\s*-", inspect.getsource(mod)), mod.__name__


# ── 이자·배당·세금 ──────────────────────────────────────────────────────────


def test_cash_interest_and_tax_krw():
    td = _days(periods=800)
    price = pd.Series(100.0, index=td)
    fx = pd.Series(1000.0, index=td)
    val = _steps(price, fx=fx, rate=0.0, n=12)[-1].date
    price.loc[val:] = 200.0  # 평가일에 두 배
    steps = _steps(price, fx=fx, rate=0.0, n=12)
    r = dca.simulate(steps, "D0", 300_000.0, krw=True)
    usd_in = 300_000 / 1000 * 0.999 * 12
    shares = usd_in / 1.0007 / 100
    sale = shares * 200 * 0.9993
    gain = sale * 1000 - usd_in * 1000
    tax = 0.22 * max(gain - 2_500_000, 0)
    assert r["tax"] == pytest.approx(tax)
    assert r["final"] == pytest.approx(sale * 1000 * 0.999 - tax)


def test_dividends_go_to_cash():
    td = _days(periods=800)
    price = pd.Series(100.0, index=td)
    steps0 = _steps(price, rate=0.0, n=3)
    ex = steps0[1].date - pd.Timedelta(days=7)
    ex = td[td <= ex][-1]
    div = pd.Series({ex: 1.0})
    steps = _steps(price, rate=0.0, n=3, div=div)
    r = dca.simulate(steps, "D0", 300.0)
    r0 = dca.simulate(steps0, "D0", 300.0)
    held = 300 / 1.0007 / 100
    assert r["final"] - r0["final"] == pytest.approx(held * 1.0 * 0.85)


def test_seal_window_stops():
    from scripts import d1_experiments as dx

    td = pd.bdate_range("2011-01-03", "2021-12-31")
    d = {"days": td, "price": pd.Series(1.0, index=td), "fx": None, "rate": pd.Series(0.0, index=td),
         "div": pd.Series(dtype=float), "facts": dca.signal_facts(pd.Series(1.0, index=td)), "seal": pd.Timestamp("2021-06-30")}
    with pytest.raises(SystemExit):
        dx.window(d, pd.Period("2011-12", "M"))
